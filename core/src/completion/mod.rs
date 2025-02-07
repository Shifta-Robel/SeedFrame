use async_trait::async_trait;

use crate::embeddings::Embedder;

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

pub(crate) type MessageHistory = Vec<Message>;

#[derive(Debug, Clone)]
pub enum CompletionError {
    ProviderError(String),
    RequestError(String),
    ParseError(String)
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
    ) -> Result<Message, CompletionError>;
}

pub struct Client<M: CompletionModel> {
    completion_model: M,
    history: MessageHistory,

    // common prompt parameters
    temperature: f64,
    max_tokens: usize,

    embedders: Vec<Embedder>,
}

impl<M: CompletionModel> Client<M> {
    pub fn new(completion_model: M, preamble: String, temperature: f64, max_tokens: usize, embedders: Vec<Embedder>) -> Self {
        Self {
            completion_model,
            history: vec![Message::Preamble(preamble)],
            embedders,
            temperature,
            max_tokens,
        }
    }


    /// Prompt the LLM and get a response.
    /// The response will be stored in the client's history
    pub async fn prompt(&mut self, prompt: &str) -> Result<Message, CompletionError> {
        let response = self
            .send_prompt(prompt, &self.history, self.temperature, self.max_tokens)
            .await;

        if let Ok(ref response) = response {
            self.history.push(response.clone());
        }

        response
    }

    /// Prompt the LLM with a custom history, and get a response.
    /// Response won't be stored in the client's history
    pub async fn one_shot(
        &mut self,
        prompt: &str,
        history: Option<MessageHistory>,
    ) -> Result<Message, CompletionError> {
        self.send_prompt(
            prompt,
            &history.unwrap_or(self.history.clone()),
            self.temperature,
            self.max_tokens,
        )
        .await
    }

    async fn send_prompt(
        &self,
        prompt: &str,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<Message, CompletionError> {
        let context = self.get_context(prompt).await;
        let message_with_context =
            Message::User(format!("{prompt}\n\n<context>\n{context}\n</context>\n"));
        let response = self
            .completion_model
            .send(message_with_context, history, temperature, max_tokens)
            .await;

        response
    }

    async fn get_context(&self, prompt: &str) -> String {
        let mut context = String::new();
        for embedder in self.embedders.iter() {
            let query_results = embedder.query(prompt, DEFAULT_TOP_N).await.unwrap();
            for r in query_results {
                context.push_str(&r.raw_data);
            }
        }
        context
    }
}
