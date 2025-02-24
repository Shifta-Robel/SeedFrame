// 
use crate::completion::{CompletionError, CompletionModel, Message, MessageHistory, TokenUsage};
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

pub struct DeepseekCompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
}

impl DeepseekCompletionModel {
    pub fn new(api_key: String, api_url: String, model: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            api_url,
            model,
        }
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "role", content = "content")]
#[allow(non_camel_case_types)]
pub enum DeepseekMessage {
    system(String),
    user(String),
    assistant(String),
}

impl From<Message> for DeepseekMessage {
    fn from(value: Message) -> DeepseekMessage {
        match value {
            Message::Preamble(s) => DeepseekMessage::system(s),
            Message::User(s) => DeepseekMessage::user(s),
            Message::Assistant(s) => DeepseekMessage::assistant(s),
        }
    }
}

#[async_trait]
impl CompletionModel for DeepseekCompletionModel {
    async fn send(
        &self,
        message: Message,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let mut messages = history.clone();
        messages.push(message);
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<DeepseekMessage>::into)
            .collect();

        let request_body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });

        let response = dbg!(self.client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await)
            .map_err(|e| CompletionError::RequestError(e.to_string()))?;

        if response.status().is_success() {
            let response_json: serde_json::Value = response
                .json()
                .await
                .map_err(|e| CompletionError::ParseError(e.to_string()))?;

            let response_message = response_json["choices"][0]["message"]["content"]
                .as_str()
                .ok_or(CompletionError::ParseError(
                    "Invalid response body".to_string(),
                ))?
                .to_string();
            let usage_response = &response_json["usage"];
            let usage_parse_error = CompletionError::ParseError("Failed to parse usage data from response".to_string());
            let token_usage =  TokenUsage {
                prompt_tokens: Some(usage_response["prompt_tokens"].as_u64().ok_or(usage_parse_error.clone())?),
                completion_tokens: Some(usage_response["completion_tokens"].as_u64().ok_or(usage_parse_error.clone())?),
                total_tokens: Some(usage_response["total_tokens"].as_u64().ok_or(usage_parse_error)?),
            };

            Ok((Message::Assistant(response_message), token_usage))
        } else {
            let error_message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(CompletionError::ProviderError(error_message))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_deepseek_completion_request() {
        tracing_subscriber::fmt().init();
        let api_key = std::env::var("SEEDFRAME_TEST_DEEPSEEK_KEY")
            .unwrap()
            .to_string();
        let api_url = "https://api.deepseek.com/chat/completions".to_string();
        let model = "deepseek".to_string();

        let openai_completion_model = DeepseekCompletionModel::new(api_key, api_url, model);

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

        assert!(response.clone().is_ok());

        assert!(response.clone().is_ok_and(|v| v.0 == Message::Assistant("okay".to_string())));
        assert!(response.is_ok_and(|v| matches!(v.1, TokenUsage { prompt_tokens: Some(_), completion_tokens: Some(_), total_tokens: Some(_) })));
    }
}
