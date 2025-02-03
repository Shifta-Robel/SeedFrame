use crate::completion::{CompletionError, CompletionModel, Message, MessageHistory};
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

#[derive(Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum OpenAIMessage {
    System(String),
    User(String),
    Assistant(String),
}

impl From<Message> for OpenAIMessage {
    fn from(value: Message) -> OpenAIMessage {
        match value {
            Message::Preamble(s) => OpenAIMessage::System(s),
            Message::User(s) => OpenAIMessage::User(s),
            Message::Assistant(s) => OpenAIMessage::Assistant(s),
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
