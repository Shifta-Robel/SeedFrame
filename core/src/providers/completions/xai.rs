use crate::completion::{CompletionError, CompletionModel, Message, MessageHistory, TokenUsage};
use crate::tools::ToolSet;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

pub struct XaiCompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
}

impl XaiCompletionModel {
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
pub enum XaiMessage {
    system(String),
    user(String),
    assistant(String),
}

impl From<Message> for XaiMessage {
    fn from(value: Message) -> XaiMessage {
        match value {
            Message::Preamble(s) => XaiMessage::system(s),
            Message::User(s) => XaiMessage::user(s),
            Message::Assistant(s) => XaiMessage::assistant(s),
        }
    }
}

#[async_trait]
impl CompletionModel for XaiCompletionModel {
    async fn send(
        &self,
        message: Message,
        history: &MessageHistory,
        tools: &ToolSet,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let mut messages = history.clone();
        messages.push(message);
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<XaiMessage>::into)
            .collect();

        let tools: Vec<serde_json::Value> = tools.0.iter().map(|t| t.default_serializer()).collect();
        let request_body = json!({
            "model": self.model,
            "messages": messages,
            "tools": tools,
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
    async fn simple_xai_completion_request() {
        tracing_subscriber::fmt().init();
        let api_key = std::env::var("SEEDFRAME_TEST_XAI_KEY")
            .unwrap()
            .to_string();
        let api_url = "https://api.x.ai/v1/chat/completions".to_string();
        let model = "grok-2-latest".to_string();

        let xai_completion_model = XaiCompletionModel::new(api_key, api_url, model);

        let response = xai_completion_model
            .send(
                Message::User(
                    r#"
This is a test from a software library that uses this LLM assistant.
For this test to be considered successful, reply with "okay" without the quotes, and NOTHING else.
"#
                    .to_string(),
                ),
                &vec![],
                &ToolSet(vec![]),
                0.0,
                10,
            )
            .await;

        assert!(response.clone().is_ok());

        assert!(response.clone().is_ok_and(|v| v.0 == Message::Assistant("okay".to_string())));
        assert!(response.is_ok_and(|v| matches!(v.1, TokenUsage { prompt_tokens: Some(_), completion_tokens: Some(_), total_tokens: Some(_) })));
    }
}
