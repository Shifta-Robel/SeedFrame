use crate::completion::{CompletionError, CompletionModel, Message, MessageHistory};
use crate::embeddings::model::{EmbeddingModel, ModelError};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

pub struct OpenAICompletionModel {
    api_key: String,
    api_url: String,
    model: String,
}

impl OpenAICompletionModel {
    pub fn new(api_key: String, api_url: String, model: String) -> Self {
        Self {
            api_key,
            api_url,
            model,
        }
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "role", content = "content")]
#[allow(non_camel_case_types)]
pub enum OpenAIMessage {
    system(String),
    user(String),
    assistant(String),
}

impl From<Message> for OpenAIMessage {
    fn from(value: Message) -> OpenAIMessage {
        match value {
            Message::Preamble(s) => OpenAIMessage::system(s),
            Message::User(s) => OpenAIMessage::user(s),
            Message::Assistant(s) => OpenAIMessage::assistant(s),
        }
    }
}

#[async_trait]
impl CompletionModel for OpenAICompletionModel {
    async fn send(
        &self,
        message: Message,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<Message, CompletionError> {
        let client = Client::new();

        let mut messages = history.clone();
        messages.push(message);
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<OpenAIMessage>::into)
            .collect();

        let request_body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });

        let response = client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|_| CompletionError::Undefined)?;

        if response.status().is_success() {
            let response_json: serde_json::Value = response
                .json()
                .await
                .map_err(|_| CompletionError::Undefined)?;

            let response_message = response_json["choices"][0]["message"]["content"]
                .as_str()
                .ok_or(CompletionError::Undefined)?
                .to_string();

            Ok(Message::Assistant(response_message))
        } else {
            let _error_message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(CompletionError::Undefined)
        }
    }
}

pub struct OpenAIEmbeddingModel {
    api_url: String,
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    pub data: Vec<OpenAIEmbeddingData>,
    object: String,
    model: String,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    pub embedding: Vec<f64>,
    object: String,
    index: usize,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, ModelError> {
        let client = Client::new();
        let request_body = json!({
                "input": data,
                "model": self.model,
        });
        let response = client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|_| ModelError::Undefined)?;

        if response.status().is_success() {
            let response = response
                .json::<OpenAIEmbeddingResponse>()
                .await
                .map_err(|_| ModelError::Undefined)?;

            Ok(response
                .data
                .into_iter()
                .map(|d| d.embedding)
                .flatten()
                .collect())
        } else {
            Err(ModelError::Undefined)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn simple_openai_request() {
        let api_key = std::env::var("SEEDFRAME_TEST_OPENAI_KEY")
            .unwrap()
            .to_string();
        let api_url = "https://api.openai.com/v1/chat/completions".to_string();
        let model = "gpt-4o-mini".to_string();

        let openai_completion_model = OpenAICompletionModel::new(api_key, api_url, model);

        let response = openai_completion_model
            .send(
                Message::User(
                    r#"
This is a test from a software library that uses this LLM assistant.
For this test to be considered successful, reply with "okay" without the quotes, and NOTHING else.
"#
                    .to_string(),
                ),
                &vec![],
                0.0,
                10,
            )
            .await;

        assert!(response.is_ok());

        assert!(response.is_ok_and(|v| v == Message::Assistant("okay".to_string())));
    }
}
