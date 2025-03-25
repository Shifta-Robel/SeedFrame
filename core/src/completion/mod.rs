use async_trait::async_trait;
use serde::Serializer;
use tracing::info;

use crate::{
    embeddings::Embedder,
    tools::{ExecutionStrategy, ToolCall, ToolResponse, ToolSet, ToolSetError},
    vector_store::VectorStoreError,
};

/// Message that'll be sent in Completions
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System prompt
    Preamble(String),
    /// Message sent by the user
    User {
        content: String,
        tool_responses: Option<Vec<ToolResponse>>,
    },
    /// Response from the assistant
    Assistant {
        content: String,
        tool_calls: Option<Vec<ToolCall>>,
    },
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

pub(crate) type MessageHistory = Vec<Message>;

#[derive(Debug, Clone)]
pub enum CompletionError {
    ProviderError(String),
    RequestError(String),
    ParseError(String),
    FailedContextFetch(VectorStoreError),
}

impl From<VectorStoreError> for CompletionError {
    fn from(value: VectorStoreError) -> Self {
        Self::FailedContextFetch(value)
    }
}

const DEFAULT_TOP_N: usize = 1;

#[async_trait]
pub trait CompletionModel {
    /// Send message to LLM and get a replay
    async fn send(
        &self,
        message: Message,
        history: &MessageHistory,
        tools: &ToolSet,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError>;
}

pub struct Client<M: CompletionModel> {
    completion_model: M,
    history: MessageHistory,

    // common prompt parameters
    temperature: f64,
    max_tokens: usize,
    tools: Box<ToolSet>,

    embedders: Vec<Embedder>,
    token_usage: TokenUsage,
}

pub struct PromptBuilder<'a, M: CompletionModel> {
    prompt: String,
    client: &'a mut Client<M>,
    execute_tools: bool,
    no_tools: bool,
    append_tool_response: bool,
    one_shot: (bool, Option<MessageHistory>),
}

impl<'a, M: CompletionModel> PromptBuilder<'a, M> {
    fn new(client:&'a mut Client<M>, prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            client,
            execute_tools: true,
            no_tools: false,
            append_tool_response: false,
            one_shot: (false, None)
        }
    }

    /// Execute the tool calls if the LLM responds with a Tool call request, `true` by default
    pub fn execute_tools(mut self, execute: bool) -> Self {
        self.execute_tools = execute;
        self
    }

    /// Don't send tool definitions with the prompt, `false` by default
    pub fn no_tools(mut self, no_tools: bool) -> Self {
        self.no_tools = no_tools;
        self
    }

    /// Create a `Message::User` with the tool reponses and append it to the client history, `false` by default
    pub fn append_tool_response(mut self, append: bool) -> Self {
        self.append_tool_response = append;
        self
    }


    /// Prompt the LLM with a custom history, and get a response.
    /// Response won't be stored in the client's history
    pub fn one_shot(mut self, one_shot: bool, history: Option<MessageHistory>) -> Self {
        self.one_shot = (one_shot, history);
        self
    }

    /// Sends the prompt to the LLM
    pub async fn send(self) -> Result<Message, CompletionError> {
        let tools = if self.no_tools {
            &Box::new(ToolSet(vec![], ExecutionStrategy::BestEffort))
        }else {
            &self.client.tools
        };
        let history = if self.one_shot.0 {
            &self.one_shot.1.unwrap_or(vec![])
        }else {
            &self.client.history
        };
        let (mut response, token_usage) = self.client
            .send_prompt(
                &self.prompt,
                history,
                tools,
                self.client.temperature,
                self.client.max_tokens,
            )
            .await?;

        self.client.history.push(Message::User {
            content: self.prompt.clone(),
            tool_responses: None,
        });
        self.client.history.push(response.clone());

        self.client.update_token_usage(&token_usage);
        if token_usage.total_tokens.is_some() {
            info!(
                "Prompt used up: {:?} tokens, Total tokens used: {:?}",
                token_usage.total_tokens, self.client.token_usage.total_tokens
            );
        }

        if self.execute_tools {
            if let Message::Assistant { content: _, tool_calls: Some(calls) } = response {
                let values = self.client.run_tools(Some(&calls)).await.unwrap();
                response = Message::User {
                    content: "".to_owned(),
                    tool_responses: Some(values.clone()),
                }
            }
        }

        if self.append_tool_response {
            self.client.append_history(&[response.clone()]);
        }

        Ok(response)
    }
}

impl<M: CompletionModel> Client<M> {
    pub fn new(
        completion_model: M,
        preamble: String,
        temperature: f64,
        max_tokens: usize,
        embedders: Vec<Embedder>,
        tools: ToolSet,
    ) -> Self {
        Self {
            completion_model,
            history: vec![Message::Preamble(preamble)],
            embedders,
            tools: Box::new(tools),
            temperature,
            max_tokens,
            token_usage: TokenUsage::default(),
        }
    }

    /// Clear conversation history while maintaining premble
    pub fn clear_history(&mut self) {
        self.history.retain(|m| matches!(m, Message::Preamble(_)));
    }

    pub fn load_history(&mut self, history: MessageHistory) {
        self.history = history;
    }

    pub fn export_history(&self) -> &MessageHistory {
        &self.history
    }

    pub fn append_history(&mut self, messages: &[Message]) {
        messages.iter().for_each(|m| self.history.push(m.clone()));
    }

    /// Creates a `PromptBuilder` instance .
    pub fn prompt(&mut self, prompt: impl Into<String>) -> PromptBuilder<M> {
        PromptBuilder::new(self, prompt)
    }

    pub async fn run_tools(
        &self,
        calls: Option<&[ToolCall]>,
    ) -> Result<Vec<ToolResponse>, ToolSetError> {
        let calls = calls.unwrap_or({
            let last = self
                .history
                .last()
                .ok_or(ToolSetError::EmptyMessageHistory)?;
            if let Message::Assistant {
                content: _,
                tool_calls: Some(tcs),
            } = last
            {
                tcs
            } else {
                Err(ToolSetError::LastMessageNotAToolCall)?
            }
        });

        let mut values = vec![];
        match self.tools.1 {
            ExecutionStrategy::FailEarly => {
                for call in calls {
                    values.push(
                        self.tools
                            .call(&call.id, &call.name, &call.arguments)
                            .await?,
                    );
                }
            }
            ExecutionStrategy::BestEffort => {
                for call in calls {
                    let tr = self.tools.call(&call.id, &call.name, &call.arguments).await;
                    if let Ok(v) = tr {
                        values.push(v);
                    }
                }
            }
        }

        Ok(values)
    }

    fn update_token_usage(&mut self, usage: &TokenUsage) {
        self.token_usage.prompt_tokens =
            combine_options(self.token_usage.prompt_tokens, usage.prompt_tokens);
        self.token_usage.completion_tokens =
            combine_options(self.token_usage.completion_tokens, usage.completion_tokens);
        self.token_usage.total_tokens =
            combine_options(self.token_usage.total_tokens, usage.total_tokens);
    }

    async fn send_prompt(
        &self,
        prompt: &str,
        history: &MessageHistory,
        tools: &ToolSet,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let context = self.get_context(prompt).await?;
        let message_with_context = Message::User {
            content: format!("{prompt}\n\n<context>\n{context}\n</context>\n"),
            tool_responses: None,
        };
        self.completion_model
            .send(
                message_with_context,
                history,
                tools,
                temperature,
                max_tokens,
            )
            .await
    }

    async fn get_context(&self, prompt: &str) -> Result<String, CompletionError> {
        let mut context = String::new();
        for embedder in self.embedders.iter() {
            let query_results = embedder.query(prompt, DEFAULT_TOP_N).await?;
            for r in query_results {
                context.push_str(&r.raw_data);
            }
        }
        Ok(context)
    }
}

pub fn serialize_user<S>(
    content: &str,
    _tool_calls: &Option<Vec<ToolResponse>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_newtype_struct("user", &content)
}

pub fn serialize_assistant<S>(
    content: &str,
    tool_calls: &Option<Vec<ToolCall>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let combined_content = match tool_calls {
        Some(calls) => &format!("{} {:?}", content, calls),
        None => content,
    };
    serializer.serialize_newtype_struct("assistant", &combined_content)
}

fn combine_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a_val), Some(b_val)) => Some(a_val + b_val),
        _ => None,
    }
}
