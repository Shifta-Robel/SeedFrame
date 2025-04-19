use crate::completion::{
    serialize_assistant, serialize_user, Client, CompletionError, CompletionModel, Message,
    MessageHistory, TokenUsage,
};
use crate::embeddings::Embedder;
use crate::tools::{ToolCall, ToolResponse, ToolSet};
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;

const API_KEY_ENV_VAR: &str = "SEEDFRAME_XAI_API_KEY";
const URL: &str = "https://api.x.ai/v1/chat/completions";
const DEFAULT_TEMP: f64 = 1.0;
const DEFAULT_TOKENS: usize = 2400;
const DEFAULT_MODEL: &str = "grok-2-latest";

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    api_key: Option<String>,
    api_url: Option<String>,
    model: Option<String>,
}

pub struct XaiCompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
}

impl XaiCompletionModel {
    #[must_use]
    pub fn new(json_config: Option<&str>) -> Self {
        let (api_key_var, api_url, model) = if let Some(json) = json_config {
            let config: ModelConfig = serde_json::from_str(json).unwrap();
            (
                config.api_key.unwrap_or(API_KEY_ENV_VAR.to_string()),
                config.api_url.unwrap_or(URL.to_string()),
                config.model.unwrap_or(DEFAULT_MODEL.to_string()),
            )
        } else {
            (
                API_KEY_ENV_VAR.to_string(),
                URL.to_string(),
                DEFAULT_MODEL.to_string(),
            )
        };
        let api_key = std::env::var(api_key_var.clone())
            .unwrap_or_else(|_| panic!("Failed to fetch env var `{api_key_var}` !"));
        Self {
            api_key,
            api_url,
            client: reqwest::Client::new(),
            model,
        }
    }
}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "role", content = "content")]
#[allow(non_camel_case_types)]
enum XaiMessage {
    system(String),
    #[serde(serialize_with = "serialize_user")]
    user {
        content: String,
        tool_responses: Option<Vec<ToolResponse>>,
    },
    #[serde(serialize_with = "serialize_assistant")]
    assistant {
        content: String,
        tool_calls: Option<Vec<ToolCall>>,
    },
}

impl From<Message> for XaiMessage {
    fn from(value: Message) -> XaiMessage {
        match value {
            Message::Preamble(s) => XaiMessage::system(s),
            Message::User {
                content,
                tool_responses,
            } => Self::user {
                content,
                tool_responses,
            },
            Message::Assistant {
                content,
                tool_calls,
            } => Self::assistant {
                content,
                tool_calls,
            },
        }
    }
}

#[allow(refining_impl_trait)]
#[async_trait]
impl CompletionModel for XaiCompletionModel {
    fn build_client(
        self,
        preamble: impl AsRef<str>,
        embedder_instances: Vec<Embedder>,
        tools: ToolSet,
    ) -> Client<Self> {
        Client::new(
            self,
            preamble,
            DEFAULT_TEMP,
            DEFAULT_TOKENS,
            embedder_instances,
            tools,
        )
    }
    async fn send(
        &mut self,
        message: Message,
        history: &MessageHistory,
        tools: Option<&ToolSet>,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let mut messages = history.clone();
        messages.push(message);
        let messages: Vec<_> = messages.into_iter().map(Into::<XaiMessage>::into).collect();

        let mut request_body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });

        if let Some(tools) = tools {
            let tools_serialized: Vec<serde_json::Value> =
                tools.0.iter().map(|t| t.default_serializer()).collect();
            if let Some(obj) = request_body.as_object_mut() {
                obj.insert(
                    "tools".to_string(),
                    serde_json::Value::Array(tools_serialized),
                );
            }
        }

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
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
            let tool_calls_json = response_json["choices"][0]["message"]["tool_calls"]
                .as_array()
                .unwrap();
            let mut tool_calls: Option<Vec<ToolCall>> = if tool_calls_json.is_empty() {
                None
            } else {
                Some(vec![])
            };
            for tc in tool_calls_json {
                let id = tc["id"].as_str().unwrap().to_string();
                let name = tc["function"]["name"].as_str().unwrap().to_string();
                let arguments = tc["function"]["arguments"].clone().to_string();
                tool_calls.as_mut().unwrap().push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            let usage_response = &response_json["usage"];
            let usage_parse_error =
                CompletionError::ParseError("Failed to parse usage data from response".to_string());
            let token_usage = TokenUsage {
                prompt_tokens: Some(
                    usage_response["prompt_tokens"]
                        .as_u64()
                        .ok_or(usage_parse_error.clone())?,
                ),
                completion_tokens: Some(
                    usage_response["completion_tokens"]
                        .as_u64()
                        .ok_or(usage_parse_error.clone())?,
                ),
                total_tokens: Some(
                    usage_response["total_tokens"]
                        .as_u64()
                        .ok_or(usage_parse_error)?,
                ),
            };

            Ok((
                Message::Assistant {
                    content: response_message,
                    tool_calls,
                },
                token_usage,
            ))
        } else {
            let status = response.status();
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error (failed to read response body)".to_string());

            Err(CompletionError::ProviderError(status.into(), error_msg))?
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn simple_xai_completion_request() {
        tracing_subscriber::fmt().init();
        let model = "grok-2-latest".to_string();

        let mut xai_completion_model = XaiCompletionModel::new(None, None, model);

        let response = xai_completion_model
            .send(
                Message::User {
                    content: r#"
This is a test from a software library that uses this LLM assistant.
For this test to be considered successful, reply with "okay" without the quotes, and NOTHING else.
"#
                    .to_string(),
                    tool_responses: None,
                },
                &vec![],
                None,
                0.0,
                10,
            )
            .await;

        assert!(response.clone().is_ok());

        assert!(response.clone().is_ok_and(|v| v.0
            == Message::Assistant {
                content: "okay".to_string(),
                tool_calls: None
            }));
        assert!(response.is_ok_and(|v| matches!(
            v.1,
            TokenUsage {
                prompt_tokens: Some(_),
                completion_tokens: Some(_),
                total_tokens: Some(_)
            }
        )));
    }
}
