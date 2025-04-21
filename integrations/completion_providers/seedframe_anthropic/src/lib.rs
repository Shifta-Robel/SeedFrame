use async_trait::async_trait;
use seedframe::completion::{Client, CompletionError, CompletionModel, Message, TokenUsage};
use seedframe::embeddings::Embedder;
use seedframe::tools::{ToolCall, ToolResponse, ToolSet};
use serde::{Deserialize, Serialize};
use serde_json::json;

const API_KEY_ENV_VAR: &str = "SEEDFRAME_ANTHROPIC_API_KEY";
const URL: &str = "https://api.anthropic.com/v1/messages ";
const DEFAULT_MODEL: &str = "claude-3-7-sonnet-20250219";
const DEFAULT_TEMP: f64 = 1.0;
const DEFAULT_TOKENS: usize = 1023;

mod utils;
type MessageHistory = Vec<Message>;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    api_key: Option<String>,
    api_url: Option<String>,
    model: Option<String>,
}

/// Implementation of Seedframe's `CompletionModel` trait for [Anthropic](https://anthropic.com).
///
/// This type is primarily designed to be used through the `#[client]` macro
/// rather than being instantiated directly. The macro will handle the
/// configuration parsing and model initialization.
///
/// # Supported Configuration
///
/// The model accepts the following configuration parameters:
///
/// - `model`: String identifier for the model to use
/// - `api_key_var`: Environment variable name containing the API key
/// - `api_url`: Custom API endpoint URL
///
/// All of the are optional so the config can be left altogeather or parts of it could be specified
///
/// # Examples
///
/// Usage with the `client` macro:
/// ```rust,no_run
/// use seedframe_anthropic::AnthropicCompletionModel;
///
/// #[client(
///     provider = "AnthropicCompletionModel",
///     config = r#"{
///       "model": "claude-3-7-sonnet-20250219",
///       "api_key_var": "ENV_VAR",
///       "api_url": "https://api.anthropic.com/v1/messages"
///     }"#
/// )]
/// struct AnthropicClient;
/// ```
/// # Error Handling
///
/// When used with the `client` macro:
/// - Invalid config json will result in a compile-time error
/// - Unknown fields in configuration will be rejected
/// - Missing API keys at runtime will result in errors
pub struct AnthropicCompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
    system: Option<String>,
}

impl AnthropicCompletionModel {
    /// Creates a new `AnthropicCompletionModel` instance with optional configuration.
    ///
    /// # Parameters
    /// - `config_json`: Optional JSON string containing configuration
    ///
    /// # Returns
    /// A new instance of `AnthropicCompletionModel`
    ///
    /// # Panics
    /// This function will panic if:
    /// - The provided JSON is malformed and cannot be parsed
    /// - The JSON contains unknown fields
    /// - Required environment variables are not set
    #[must_use]
    pub fn new(config_json: Option<&str>) -> Self {
        let (api_key_var, api_url, model) = if let Some(json) = config_json {
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
        let api_key = std::env::var(api_key_var).unwrap();
        Self {
            api_key,
            api_url,
            client: reqwest::Client::new(),
            model,
            system: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
#[serde(tag = "role", content = "content")]
#[allow(non_camel_case_types)]
pub(crate) enum AnthropicMessage {
    user(Vec<ContentBlock>),
    assistant(Vec<ContentBlock>),
}
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Debug)]
#[serde(tag = "type")]
pub(crate) enum ContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        data: String,
    },
    ToolUse(ToolCall),
    ToolResult(ToolResponse),
}

impl From<Message> for AnthropicMessage {
    fn from(value: Message) -> Self {
        match value {
            Message::Preamble(_) => Self::user(vec![ContentBlock::Text {
                text: String::new(),
            }]),
            Message::User {
                content,
                tool_responses,
            } => {
                let mut out: Vec<ContentBlock> = Vec::new();
                if !content.is_empty() {
                    let vals = utils::parse_content_blocks(&content);
                    vals.iter().for_each(|v| out.push(v.clone()));
                }
                if let Some(tools) = tool_responses {
                    tools
                        .iter()
                        .for_each(|t| out.push(ContentBlock::ToolResult(t.clone())));
                }
                Self::user(out)
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let mut out = Vec::new();
                if !content.is_empty() {
                    let vals = utils::parse_content_blocks(&content);
                    vals.iter().for_each(|v| out.push(v.clone()));
                }
                if let Some(tools) = tool_calls {
                    tools
                        .iter()
                        .for_each(|t| out.push(ContentBlock::ToolUse(t.clone())));
                }
                Self::user(out)
            }
        }
    }
}

#[allow(refining_impl_trait)]
#[async_trait]
impl CompletionModel for AnthropicCompletionModel {
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
    #[allow(clippy::too_many_lines)]
    async fn send(
        &mut self,
        message: Message,
        history: &MessageHistory,
        tools: Option<&ToolSet>,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError> {
        let mut messages = history.clone();
        if let Some(Message::Preamble(p)) = messages.first() {
            self.system = Some(p.clone());
            messages.remove(0);
        }
        messages.push(message);
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<AnthropicMessage>::into)
            .collect();

        let mut request_body = json!({
            "store": true,
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

        if let Some(preamble) = &self.system {
            if let Some(obj) = request_body.as_object_mut() {
                obj.insert(
                    "system".to_string(),
                    serde_json::Value::String(preamble.to_string()),
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

            let mut content: Vec<String> = vec![];
            let mut tool_calls: Vec<ToolCall> = vec![];
            let () = &response_json["content"]
                .as_array()
                .unwrap()
                .iter()
                .for_each(|c| match c["type"].as_str() {
                    Some("text") => {
                        content.push(c["text"].as_str().unwrap().to_string());
                    }
                    Some("thinking") => {
                        let thought = c["thinking"].as_str().unwrap().to_string();
                        let sig = c["signature"].as_str().unwrap().to_string();
                        content.push(format!(
                            "<sf_thinking>{thought}</sf_sig>{sig}</sf_thinking>"
                        ));
                    }
                    Some("tool_use") => {
                        let id = c["id"].as_str().unwrap().to_string();
                        let name = c["name"].as_str().unwrap().to_string();
                        let arguments = c["input"].as_str().unwrap().to_string();
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    Some("redacted_thinking") => {
                        let data = c["data"].as_str().unwrap().to_string();
                        content.push(format!("<sf_r_thinking>{data}</sf_r_thinking>"));
                    }
                    _ => {}
                });
            let content: String = content.join("");
            let usage_response = &response_json["usage"];
            let usage_parse_error =
                CompletionError::ParseError("Failed to parse usage data from response".to_string());
            let input_tokens = usage_response["input_tokens"]
                .as_u64()
                .ok_or(usage_parse_error.clone())?;
            let output_tokens = usage_response["output_tokens"]
                .as_u64()
                .ok_or(usage_parse_error.clone())?;
            let token_usage = TokenUsage {
                prompt_tokens: Some(input_tokens),
                completion_tokens: Some(output_tokens),
                total_tokens: Some(input_tokens + output_tokens),
            };

            let tool_calls = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            };
            Ok((
                Message::Assistant {
                    content,
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
mod test {
    use seedframe::completion::Message;

    use crate::{AnthropicMessage, ContentBlock};

    #[test]
    fn test_proper_message_conversion() {
        let st = "start<sf_thinking>think1</sf_sig>make</sf_thinking>middle<sf_r_thinking>think2</sf_r_thinking>end";
        let crate_messages = Message::User {
            content: String::from(st),
            tool_responses: None,
        };
        let converted = Into::<AnthropicMessage>::into(crate_messages);
        assert_eq!(
            converted,
            AnthropicMessage::user(vec![
                ContentBlock::Text {
                    text: "start".to_string()
                },
                ContentBlock::Thinking {
                    thinking: "think1".to_string(),
                    signature: "make".to_string()
                },
                ContentBlock::Text {
                    text: "middle".to_string()
                },
                ContentBlock::RedactedThinking {
                    data: "think2".to_string()
                },
                ContentBlock::Text {
                    text: "end".to_string()
                },
            ])
        );
    }
}
