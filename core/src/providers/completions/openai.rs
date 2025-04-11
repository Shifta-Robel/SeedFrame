use crate::completion::{
    default_extractor_serializer, serialize_assistant, serialize_user, Client, CompletionError, CompletionModel, Extractor, Message, MessageHistory, TokenUsage
};
use crate::embeddings::Embedder;
use crate::tools::{ToolCall, ToolResponse, ToolSet};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

const API_KEY_ENV_VAR: &str = "SEEDFRAME_OPENAI_API_KEY";
const URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_TEMP: f64 = 1.0;
const DEFAULT_TOKENS: usize = 2400;

pub struct OpenAICompletionModel {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
    model: String,
}

impl OpenAICompletionModel {
    #[must_use] pub fn new(api_key: Option<String>, api_url: Option<String>, model: String) -> Self {
        let api_key = api_key.unwrap_or(std::env::var(API_KEY_ENV_VAR).unwrap());
        let api_url = api_url.unwrap_or(URL.to_string());
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
pub enum OpenAIMessage {
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
            tools
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
                    calls
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
                        .collect()
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

        let extractor = default_extractor_serializer::<T>().map_err(|e| {
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

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CompletionError::RequestError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error (failed to read response body)".to_string());

            return Err(CompletionError::ProviderError(status.into(), error_msg));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CompletionError::ParseError(e.to_string()))?;

        let extracted_str = response_json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| CompletionError::ParseError("Missing content".to_string()))?;

        let extracted: T = serde_json::from_str(extracted_str)
            .map_err(|e| CompletionError::ParseError(e.to_string()))?;

        Ok(extracted)
    }
}

#[cfg(test)]
mod tests {
    use std::any::{Any, TypeId};

    use dashmap::DashMap;
    use serde_json::Value;

    use crate::tools::{ExecutionStrategy, Tool, ToolArg, ToolError};
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn simple_openai_completion_request() {
        let model = "gpt-4o-mini".to_string();

        let mut openai_completion_model = OpenAICompletionModel::new(None, None, model);

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
        let model = "gpt-4o-mini".to_string();

        let mut openai_completion_model = OpenAICompletionModel::new(None, None, model);
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
            async fn call(&self, args: &str, _states: &DashMap<TypeId, Box<dyn Any + Send + Sync>>) -> Result<Value, ToolError> {
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
            async fn call(&self, args: &str, _states: &DashMap<TypeId, Box<dyn Any + Send + Sync>>) -> Result<Value, ToolError> {
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
