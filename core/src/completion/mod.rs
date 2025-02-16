use async_trait::async_trait;
use tracing::info;

use crate::{embeddings::Embedder, vector_store::VectorStoreError};

/// Message that'll be sent in Completions
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Message {
    /// System prompt
    Preamble(String),
    /// Message sent by user
    User(String),
    /// Response from the assistant
    Assistant(String),
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
    FailedContextFetch(VectorStoreError)
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

    embedders: Vec<Embedder>,
    token_usage: TokenUsage
}

impl<M: CompletionModel> Client<M> {
    pub fn new(completion_model: M, preamble: String, temperature: f64, max_tokens: usize, embedders: Vec<Embedder>) -> Self {
        Self {
            completion_model,
            history: vec![Message::Preamble(preamble)],
            embedders,
            temperature,
            max_tokens,
            token_usage: TokenUsage::default()
        }
    }

    pub fn clear_history(&mut self) {
        self.history.retain(|m| matches!(m, Message::Preamble(_)));
    }
    
    pub fn load_history(&mut self, history: MessageHistory) {
        self.history =  history;
    }

    pub fn export_history(&self) -> &MessageHistory {
        &self.history
    }

    /// Prompt the LLM and get a response.
    /// The response will be stored in the client's history
    pub async fn prompt(&mut self, prompt: &str) -> Result<Message, CompletionError> {
        let (response, token_usage) = self
            .send_prompt(prompt, &self.history, self.temperature, self.max_tokens)
            .await?;

        self.history.push(response.clone());
        self.update_token_usage(&token_usage);
        if token_usage.total_tokens.is_some() {
            info!("Prompt used up: {:?} tokens, Total tokens used: {:?}", token_usage.total_tokens, self.token_usage.total_tokens);
        }

        Ok(response)
    }

    fn update_token_usage(&mut self, usage: &TokenUsage) {
        self.token_usage.prompt_tokens = combine_options(self.token_usage.prompt_tokens, usage.prompt_tokens);
        self.token_usage.completion_tokens = combine_options(self.token_usage.completion_tokens, usage.completion_tokens);
        self.token_usage.total_tokens = combine_options(self.token_usage.total_tokens, usage.total_tokens);
    }

    /// Prompt the LLM with a custom history, and get a response.
    /// Response won't be stored in the client's history
    pub async fn one_shot(
        &mut self,
        prompt: &str,
        history: Option<MessageHistory>,
    ) -> Result<Message, CompletionError> {
        let (response, token_usage) = self
            .send_prompt( prompt, &history.unwrap_or(self.history.clone()), self.temperature, self.max_tokens).await?;

        self.update_token_usage(&token_usage);
        if token_usage.total_tokens.is_some() {
            info!("Prompt used up: {:?} tokens, Total tokens used: {:?}", token_usage.total_tokens, self.token_usage.total_tokens);
        }

        Ok(response)
    }

    async fn send_prompt(
        &self,
        prompt: &str,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let context = self.get_context(prompt).await?;
        let message_with_context =
            Message::User(format!("{prompt}\n\n<context>\n{context}\n</context>\n"));
        self
            .completion_model
            .send(message_with_context, history, temperature, max_tokens)
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

fn combine_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a_val), Some(b_val)) => Some(a_val + b_val),
        _ => None,
    }
}
