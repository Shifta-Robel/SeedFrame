use crate::completion::{
    default_extractor_serializer, serialize_assistant, serialize_user, Client, CompletionError,
    CompletionModel, Extractor, Message, MessageHistory, TokenUsage,
};
use crate::embeddings::Embedder;
use crate::tools::{ToolCall, ToolResponse, ToolSet};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, error, info, instrument};

const API_KEY_ENV_VAR: &str = "SEEDFRAME_OPENAI_API_KEY";
const URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_TEMP: f64 = 1.0;
const DEFAULT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_TOKENS: usize = 2400;

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    api_key: Option<String>,
    api_url: Option<String>,
    model: Option<String>,
}

pub struct OpenAICompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
}

impl OpenAICompletionModel {
    #[instrument]
    #[must_use]
    pub fn new(json_config: Option<&str>) -> Self {
        let (api_key_var, api_url, model) = if let Some(json) = json_config {
            let config = match serde_json::from_str::<ModelConfig>(json) {
                Ok(config) => config,
                Err(e) => {
                    let e = format!("Failed to deserialize json config: {e}");
                    error!(e);
                    panic!("{e}");
                }
            };
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
        let api_key = match std::env::var(&api_key_var) {
            Ok(key) => key,
            Err(e) => {
                let e = format!("Failed to fetch env var `{api_key_var}`!, {e}");
                error!(e);
                panic!("{e}");
            }
        };
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
enum OpenAIMessage {
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

impl From<Message> for OpenAIMessage {
    fn from(value: Message) -> OpenAIMessage {
        match value {
            Message::Preamble(s) => OpenAIMessage::system(s),
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
impl CompletionModel for OpenAICompletionModel {
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
    #[instrument(
        skip(self, history, tools, temperature),
        fields(
            history_len = history.len(),
            tools = tools.is_some())
    )]
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
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<OpenAIMessage>::into)
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
                info!(
                    tool_count = tools_serialized.len(),
                    "Including tools in request"
                );
                obj.insert(
                    "tools".to_string(),
                    serde_json::Value::Array(tools_serialized),
                );
            }
        }

        debug!(request_body = ?request_body, "Sending request to OpenAI");

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!(error = ?e, "Request failed");
                CompletionError::RequestError(e.to_string())
            })?;

        let status = response.status();
        debug!(%status, "Received API response");

        if status.is_success() {
            let response_json: serde_json::Value = response.json().await.map_err(|e| {
                error!(error = ?e, "Failed to parse response JSON");
                // error!("{}",format!("Failed to parse response as json: {:?}",e));
                CompletionError::ParseError(e.to_string())
            })?;

            let resp_msg_json = &response_json["choices"][0]["message"]["content"];
            let mut response_message = String::new();
            if !resp_msg_json.is_null() {
                response_message = resp_msg_json
                    .as_str()
                    .ok_or(CompletionError::ParseError(
                        "Invalid response body".to_string(),
                    ))?
                    .to_string();
            }

            let tool_calls: Option<Vec<ToolCall>> = response_json["choices"][0]["message"]
                ["tool_calls"]
                .as_array()
                .filter(|calls| !calls.is_empty())
                .map(|calls| {
                    let count = calls.len();
                    let result = calls
                        .iter()
                        .map(|tc| {
                            let id = tc["id"].as_str().unwrap().to_string();
                            let name = tc["function"]["name"].as_str().unwrap().to_string();
                            let arguments = tc["function"]["arguments"].clone().to_string();
                            ToolCall {
                                id,
                                name,
                                arguments,
                            }
                        })
                        .collect();
                    info!(tool_call_count = count, "Parsed tool calls");
                    result
                });

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

            info!(
                prompt_tokens = token_usage.prompt_tokens,
                completion_tokens = token_usage.completion_tokens,
                total_tokens = token_usage.total_tokens,
                "Token usage recorded"
            );

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

            error!(
                status = %status,
                error = %error_msg,
                "API returned error response"
            );

            Err(CompletionError::ProviderError(status.into(), error_msg))?
        }
    }

    #[instrument(
        skip(self, history, temperature),
        fields(history_len = history.len())
    )]
    async fn extract<T: Extractor>(
        &mut self,
        message: Message,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<T, CompletionError> {
        let mut messages = history.clone();
        messages.push(message);
        let messages: Vec<_> = messages
            .into_iter()
            .map(Into::<OpenAIMessage>::into)
            .collect();
        info!(
            message_count = messages.len(),
            "Preparing extraction request"
        );

        let extractor = default_extractor_serializer::<T>().map_err(|e| {
            error!(error = ?e, "Failed to serialize extractor");
            CompletionError::ParseError(format!("Failed to serialize extrator: {e}"))
        })?;

        let request_body = json!({
            "store": true,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": extractor,
        });
        debug!(request_body = ?request_body, "Sending extraction request");

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CompletionError::RequestError(e.to_string()))?;

        let status = response.status();
        debug!(%status, "Received extraction response");

        if !response.status().is_success() {
            let status = response.status();
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error (failed to read response body)".to_string());

            error!(
                status = %status,
                error = %error_msg,
                "Extraction API returned error"
            );
            return Err(CompletionError::ProviderError(status.into(), error_msg));
        }

        let response_json: serde_json::Value = response.json().await.map_err(|e| {
            error!(error = ?e, "Failed to parse extraction response JSON");
            CompletionError::ParseError(e.to_string())
        })?;

        let extracted_str = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                error!("Missing content in extraction response");
                CompletionError::ParseError("Missing content".to_string())
            })?;

        let extracted: T = serde_json::from_str(extracted_str)
            .map_err(|e| {
                error!(error = ?e, raw_response = %extracted_str, "Failed to deserialize extracted content");
                CompletionError::ParseError(e.to_string())})?;

        info!(
            extractor_type = std::any::type_name::<T>(),
            "Successfully extracted data"
        );
        Ok(extracted)
    }
}

#[cfg(test)]
mod tests {
    use std::any::{Any, TypeId};

    use dashmap::DashMap;
    use serde_json::Value;

    use super::*;
    use crate::tools::{ExecutionStrategy, Tool, ToolArg, ToolError};

    #[tokio::test]
    #[ignore]
    async fn simple_openai_completion_request() {
        let mut openai_completion_model = OpenAICompletionModel::new(None);

        let response = openai_completion_model
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

        assert!(response.is_ok());

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
    #[tokio::test]
    #[ignore]
    async fn openai_toolcall_test() {
        tracing_subscriber::fmt().init();
        let mut openai_completion_model = OpenAICompletionModel::new(None);
        let response = openai_completion_model
            .send(
                Message::User {
                    content: "Tell me a joke in Farsi".to_string(),
                    tool_responses: None,
                },
                &vec![],
                Some(&get_tools()),
                0.0,
                1000,
            )
            .await;

        assert!(response.is_ok());
        assert!(matches!(
            response.unwrap().0,
            Message::Assistant {
                content: _,
                tool_calls: Some(_)
            }
        ));
    }

    fn get_tools() -> ToolSet {
        struct JokeTool {
            args: Vec<ToolArg>,
        }
        struct PoemTool {
            args: Vec<ToolArg>,
        }
        impl JokeTool {
            pub fn new() -> Self {
                Self {
                    args: vec![ToolArg::new::<String>(
                        "lang",
                        "language to tell the joke in",
                    )],
                }
            }
        }
        impl PoemTool {
            pub fn new() -> Self {
                Self {
                    args: vec![ToolArg::new::<u32>(
                        "length",
                        "how many words the poem should be",
                    )],
                }
            }
        }
        fn tell_joke(lang: &str) -> String {
            format!("a funny joke in the {lang} language")
        }
        fn tell_poem(length: u32) -> String {
            format!("a poem thats {length} words long!!")
        }

        #[async_trait]
        impl Tool for JokeTool {
            async fn call(
                &self,
                args: &str,
                _states: &DashMap<TypeId, Box<dyn Any + Send + Sync>>,
            ) -> Result<Value, ToolError> {
                #[derive(serde::Deserialize)]
                struct Params {
                    lang: String,
                }
                let params: Params = serde_json::from_str(args)?;
                Ok(serde_json::Value::from(tell_joke(&params.lang)))
            }
            fn name(&self) -> &str {
                "tell_joke"
            }
            fn args(&self) -> &[ToolArg] {
                &self.args
            }
            fn description(&self) -> &str {
                "Tells jokes in the given language"
            }
        }
        #[async_trait]
        impl Tool for PoemTool {
            async fn call(
                &self,
                args: &str,
                _states: &DashMap<TypeId, Box<dyn Any + Send + Sync>>,
            ) -> Result<Value, ToolError> {
                #[derive(serde::Deserialize)]
                struct Params {
                    lenght: u32,
                }
                let params: Params = serde_json::from_str(args)?;
                Ok(serde_json::Value::from(tell_poem(params.lenght)))
            }
            fn name(&self) -> &str {
                "tell_poem"
            }
            fn args(&self) -> &[ToolArg] {
                &self.args
            }
            fn description(&self) -> &str {
                "Tells poems with the given number of words"
            }
        }
        ToolSet(
            vec![Box::new(JokeTool::new()), Box::new(PoemTool::new())],
            ExecutionStrategy::FailEarly,
        )
    }
}
