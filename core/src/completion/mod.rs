use async_trait::async_trait;
use dashmap::DashMap;
use schemars::gen::SchemaSettings;
use serde::Serializer;
use serde_json::json;
use std::{
    any::{Any, TypeId},
    sync::Arc,
};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::info;

use crate::{
    embeddings::Embedder,
    tools::{ExecutionStrategy, ToolCall, ToolResponse, ToolSet, ToolSetError},
    vector_store::VectorStoreError,
};

// Default top_n context documents to query from the vector store
const DEFAULT_TOP_N: usize = 1;

/// Messages exchanged with the completion model
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// System prompt
    Preamble(String),
    /// Message sent by the user
    User {
        /// Text content of the message
        content: String,
        /// Optional tool execution results
        tool_responses: Option<Vec<ToolResponse>>,
    },
    /// Model-generated response
    Assistant {
        /// Text content of the response
        content: String,
        /// Optional requested tool calls
        tool_calls: Option<Vec<ToolCall>>,
    },
}

/// Tracks token usage statistics for model interactions
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens consumed by the prompt input
    pub prompt_tokens: Option<u64>,
    /// Tokens generated in the completion output
    pub completion_tokens: Option<u64>,
    /// Combined total of prompt and completion tokens
    pub total_tokens: Option<u64>,
}

pub(crate) type MessageHistory = Vec<Message>;

#[derive(Debug, Clone, Error)]
/// Errors that can happen during completion
pub enum CompletionError {
    /// Errors returned from the provider
    #[error("Provider error -> HTTP Status {0}: {1}")]
    ProviderError(u16, String),
    /// Error within the completion request
    #[error("RequestError: {0}")]
    RequestError(String),
    /// Error while parsing compleition reponse
    #[error("ParseError: {0}")]
    ParseError(String),
    /// Error while fetching the context from the vector store
    #[error("Failed to fetch context: {0}")]
    FailedContextFetch(#[from] VectorStoreError),
    /// Error while extracting
    #[error(transparent)]
    ExtractorError(#[from] ExtractionError),
    /// Error with tool call states
    #[error(transparent)]
    StateError(#[from] StateError),
}

/// Types that can be deserialized from model completion responses.
///
/// Requires JSON Schema generation and owned deserialization capabilities.
/// Implement this trait for types that should be extractable from model outputs.
pub trait Extractor: schemars::JsonSchema + serde::de::DeserializeOwned {}

/// Errors related to response extraction from completions
#[derive(Debug, Clone, Error)]
pub enum ExtractionError {
    #[error("Model does not support extraction")]
    ExtractionNotSupported,
}

/// Errors related to state management in the [`Client`]
#[derive(Debug, Clone, Error)]
pub enum StateError {
    /// Attempted to register state when state of this type already exists
    #[error("State with type {0} already exists on client")]
    AlreadyExists(String),
    /// Requested state type was not found in the client
    #[error("State not found")]
    NotFound,
}

/// Core trait defining the interface for completion models
#[async_trait]
pub trait CompletionModel: Send {
    /// Constructs a new [`Client`] with this model
    ///
    /// # Arguments
    /// * `preamble` - System prompt/instructions for the model
    /// * `embedder_instances` - Embedding models for context retrieval
    /// * `tools` - Collection of available tools
    fn build_client(
        self,
        preamble: impl AsRef<str>,
        embedder_instances: Vec<crate::embeddings::Embedder>,
        tools: ToolSet,
    ) -> Client<impl CompletionModel>;

    /// Sends a message to the model and returns its response
    ///
    /// # Arguments
    /// * `message` - The message to process
    /// * `history` - Conversation context
    /// * `tools` - Optional toolset for function calling
    /// * `temperature` - Sampling temperature (0.0-1.0)
    /// * `max_tokens` - Maximum tokens to be used
    async fn send(
        &mut self,
        message: Message,
        history: &MessageHistory,
        tools: Option<&ToolSet>,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<(Message, TokenUsage), CompletionError>;

    #[allow(unused)]
    /// Extracts structured data from a model response
    ///
    /// Default implementation returns [`ExtractionError::ExtractionNotSupported`]
    /// unless overridden by the model implementation.
    async fn extract<T: Extractor>(
        &mut self,
        message: Message,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<T, CompletionError> {
        Err(CompletionError::ExtractorError(
            ExtractionError::ExtractionNotSupported,
        ))
    }
}

/// Extractor for state
pub struct State<T: Send + Sync + 'static>(pub Arc<T>);

/// A client for managing interactions with a completion model, including conversation history,
/// tooling, state management, and token tracking.
///
/// The client is generic over any type implementing [`CompletionModel`] and provides:
/// - Thread-safe access to the underlying model
/// - Conversation history management
/// - Completion parameters configuration
/// - Tool integration through [`ToolSet`]
/// - Embeddings
/// - State management for tools
/// - Token usage tracking
pub struct Client<M: CompletionModel> {
    completion_model: Arc<tokio::sync::RwLock<M>>,
    /// Conversation history maintaining message context
    history: MessageHistory,
    /// Collection of available tools for the model to use
    tools: Box<ToolSet>,
    /// Embedding models for text vectorization
    embedders: Vec<Embedder>,
    /// Tracking of token usage statistics
    token_usage: TokenUsage,
    /// Type-mapped state storage for arbitrary values
    states: DashMap<TypeId, Box<dyn Any + Send + Sync>>,

    // common prompt parameters
    temperature: f64,
    max_tokens: usize,
}

/// Builder for constructing and executing completion prompts
pub struct PromptBuilder<'a, M: CompletionModel> {
    prompt: String,
    client: &'a mut Client<M>,
    execute_tools: bool,
    with_tools: bool,
    append_tool_response: bool,
    one_shot: (bool, Option<MessageHistory>),
    with_context: bool,
}

impl<'a, M: CompletionModel> PromptBuilder<'a, M> {
    fn new(client: &'a mut Client<M>, prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            client,
            execute_tools: true,
            with_tools: true,
            append_tool_response: false,
            one_shot: (false, None),
            with_context: true,
        }
    }

    /// Execute the tool calls if the LLM responds with a Tool call request, `true` by default
    #[must_use]
    pub fn execute_tools(mut self, execute: bool) -> Self {
        self.execute_tools = execute;
        self
    }

    /// Wether to send any tool definitions with the prompt, `true` by default
    #[must_use]
    pub fn with_tools(mut self, no_tools: bool) -> Self {
        self.with_tools = no_tools;
        self
    }

    /// Create a `Message::User` with the tool reponses and append it to the client history, `false` by default
    #[must_use]
    pub fn append_tool_response(mut self, append: bool) -> Self {
        self.append_tool_response = append;
        self
    }

    /// Wether to retrieve and append the context to the prompt, true by default.
    /// If true, the client's vector store will get looked up for the top matches to the prompt and
    /// the context will get appended to the prompt before being sent.
    #[must_use]
    pub fn with_context(mut self, append_context: bool) -> Self {
        self.with_context = append_context;
        self
    }

    /// Prompt the LLM with a custom history, and get a response.
    /// Response won't be stored in the client's history
    #[must_use]
    pub fn one_shot(mut self, one_shot: bool, history: Option<MessageHistory>) -> Self {
        self.one_shot = (one_shot, history);
        self
    }

    /// Extracts structured data from the model's response
    ///
    /// Handles context retrieval and message construction automatically.
    ///
    /// # Errors
    /// Returns errors from:
    /// - Context retrieval ([`VectorStoreError`])
    /// - Model execution ([`CompletionError`])
    /// - Extraction ([`ExtractionError`])
    pub async fn extract<T: Extractor>(self) -> Result<T, crate::error::Error> {
        let history = if self.one_shot.0 {
            &self.one_shot.1.unwrap_or_default()
        } else {
            &self.client.history
        };

        let retrieved_context = self.client.get_context(&self.prompt).await?;
        let context = if self.with_context {
            retrieved_context
                .map_or_else(String::new, |c| format!("\n\n<context>\n{c}\n</context>\n"))
        } else {
            String::new()
        };

        let message = Message::User {
            content: format!("{}{}", self.prompt, context),
            tool_responses: None,
        };

        let model = self.client.completion_model.clone();
        let mut guard = model.write().await;

        guard
            .extract::<T>(
                message,
                history,
                self.client.temperature,
                self.client.max_tokens,
            )
            .await
            .map_err(Into::into)
    }

    /// Sends the prompt to the LLM
    pub async fn send(self) -> Result<Message, crate::error::Error> {
        let tools = if self.with_tools && !self.client.tools.0.is_empty() {
            Some(&*self.client.tools)
        } else {
            None
        };
        let history = if self.one_shot.0 {
            &self.one_shot.1.unwrap_or_default()
        } else {
            &self.client.history
        };
        let (mut response, token_usage) = self
            .client
            .send_prompt(
                &self.prompt,
                history,
                tools,
                self.client.temperature,
                self.client.max_tokens,
                self.with_context,
            )
            .await?;

        if !self.one_shot.0 {
            self.client.history.push(Message::User {
                content: self.prompt.clone(),
                tool_responses: None,
            });
            self.client.history.push(response.clone());
        }

        self.client.update_token_usage(&token_usage);
        if token_usage.total_tokens.is_some() {
            info!(
                "Prompt used up: {:?} tokens, Total tokens used: {:?}",
                token_usage.total_tokens, self.client.token_usage.total_tokens
            );
        }

        if self.execute_tools {
            if let Message::Assistant {
                content: _,
                tool_calls: Some(calls),
            } = response.clone()
            {
                if self.one_shot.0 {
                    self.client.history.push(response);
                }
                let values = self.client.run_tools(Some(&calls)).await?;
                if self.one_shot.0 {
                    self.client.history.pop();
                }
                response = Message::User {
                    content: String::new(),
                    tool_responses: Some(values.clone()),
                };
                if self.append_tool_response && !self.one_shot.0 {
                    self.client.append_history(&[response.clone()]);
                }
            }
        }

        Ok(response)
    }
}

impl<M: CompletionModel + Send> Client<M> {
    /// Creates a new client with the specified configuration
    ///
    /// # Arguments
    /// * `completion_model` - The underlying model implementation
    /// * `preamble` - System instructions for the model
    /// * `temperature` - Sampling temperature (0.0-1.0)
    /// * `max_tokens` - Maximum response length
    /// * `embedders` - Embedding models for context retrieval
    /// * `tools` - Available tools for function calling
    pub fn new(
        completion_model: M,
        preamble: impl AsRef<str>,
        temperature: f64,
        max_tokens: usize,
        embedders: Vec<Embedder>,
        tools: ToolSet,
    ) -> Self {
        Self {
            completion_model: Arc::new(RwLock::new(completion_model)),
            history: vec![Message::Preamble(String::from(preamble.as_ref()))],
            embedders,
            tools: Box::new(tools),
            temperature,
            max_tokens,
            token_usage: TokenUsage::default(),
            states: DashMap::new(),
        }
    }

    /// Clear conversation history while maintaining premble
    pub fn clear_history(&mut self) {
        self.history.retain(|m| matches!(m, Message::Preamble(_)));
    }

    /// Replaces the current message history with the provided history
    pub fn load_history(&mut self, history: MessageHistory) {
        self.history = history;
    }

    /// Returns a reference to the current message history
    #[must_use]
    pub fn export_history(&self) -> &MessageHistory {
        &self.history
    }

    /// Appends messages to the conversation history
    pub fn append_history(&mut self, messages: &[Message]) {
        messages.iter().for_each(|m| self.history.push(m.clone()));
    }

    /// Registers new state with the client
    ///
    /// # Errors
    /// Returns [`StateError::AlreadyExists`] if state of this type is already registered
    pub fn with_state<T: Send + Sync + 'static>(self, state: T) -> Result<Self, CompletionError> {
        let type_id = state.type_id();
        if self.states.contains_key(&type_id) {
            return Err(CompletionError::StateError(StateError::AlreadyExists(
                format!("{:?}", std::any::type_name::<T>()),
            )));
        }
        self.states.insert(type_id, Box::new(Arc::new(state)));
        Ok(self)
    }

    /// Retrieves previously registered state
    ///
    /// # Errors
    /// Returns [`StateError::NotFound`] if no state of type `T` exists
    pub fn get_state<T: Send + Sync + 'static>(&self) -> Result<State<T>, StateError> {
        let boxed = self
            .states
            .get(&TypeId::of::<T>())
            .ok_or(StateError::NotFound)?;

        let arc = boxed.downcast_ref::<Arc<T>>().ok_or(StateError::NotFound)?;

        Ok(State(arc.clone()))
    }

    /// Creates a `PromptBuilder` instance .
    pub fn prompt(&mut self, prompt: impl Into<String>) -> PromptBuilder<M> {
        PromptBuilder::new(self, prompt)
    }

    /// Executes requested tool calls from the model
    ///
    /// If no calls are provided, attempts to use calls from the last assistant message.
    /// Supports different execution strategies (fail-early vs best-effort).
    ///
    /// # Errors
    /// Returns [`ToolSetError`] for:
    /// - Empty message history when calls not provided
    /// - Last message not containing tool calls
    /// - Individual tool execution failures
    pub async fn run_tools(
        &self,
        calls: Option<&[ToolCall]>,
    ) -> Result<Vec<ToolResponse>, ToolSetError> {
        let calls = calls.unwrap_or({
            let last = self
                .history
                .last()
                .ok_or(ToolSetError::EmptyMessageHistory)?;
            if let Message::Assistant {
                content: _,
                tool_calls: Some(tcs),
            } = last
            {
                tcs
            } else {
                Err(ToolSetError::LastMessageNotAToolCall)?
            }
        });

        let mut values = vec![];
        match self.tools.1 {
            ExecutionStrategy::FailEarly => {
                for call in calls {
                    values.push(
                        self.tools
                            .call(&call.id, &call.name, &call.arguments, &self.states)
                            .await?,
                    );
                }
            }
            ExecutionStrategy::BestEffort => {
                for call in calls {
                    let tr = self
                        .tools
                        .call(&call.id, &call.name, &call.arguments, &self.states)
                        .await;
                    if let Ok(v) = tr {
                        values.push(v);
                    }
                }
            }
        }

        Ok(values)
    }

    fn update_token_usage(&mut self, usage: &TokenUsage) {
        self.token_usage.prompt_tokens =
            combine_options(self.token_usage.prompt_tokens, usage.prompt_tokens);
        self.token_usage.completion_tokens =
            combine_options(self.token_usage.completion_tokens, usage.completion_tokens);
        self.token_usage.total_tokens =
            combine_options(self.token_usage.total_tokens, usage.total_tokens);
    }

    async fn send_prompt(
        &self,
        prompt: &str,
        history: &MessageHistory,
        tools: Option<&ToolSet>,
        temperature: f64,
        max_tokens: usize,
        append_context: bool,
    ) -> Result<(Message, TokenUsage), crate::error::Error> {
        let retrieved_context = self.get_context(prompt).await?;
        let context = if append_context {
            retrieved_context
                .map_or_else(String::new, |c| format!("\n\n<context>\n{c}\n</context>\n"))
        } else {
            String::new()
        };

        let message_with_context = Message::User {
            content: format!("{prompt}{context}"),
            tool_responses: None,
        };

        let model = self.completion_model.clone();
        let mut guard = model.write().await;
        guard
            .send(
                message_with_context,
                history,
                tools,
                temperature,
                max_tokens,
            )
            .await
            .map_err(crate::error::Error::from)
    }

    async fn get_context(&self, prompt: &str) -> Result<Option<String>, VectorStoreError> {
        if self.embedders.is_empty() {
            return Ok(None);
        }
        let mut context = String::new();
        for embedder in &self.embedders {
            let query_results = embedder.query(prompt, DEFAULT_TOP_N).await?;
            if query_results.is_empty() {
                return Ok(None);
            }
            for r in query_results {
                context.push_str(&r.raw_data);
            }
        }
        Ok(Some(context))
    }
}

/// Generates a JSON schema serializer for extractor types
///
/// # Errors
/// Returns `serde_json::Error` if schema serialization fails
pub fn default_extractor_serializer<'a, T: schemars::JsonSchema + serde::Deserialize<'a>>(
) -> Result<serde_json::Value, serde_json::error::Error> {
    let settings = SchemaSettings::default().with(|s| {
        s.inline_subschemas = true;
    });
    let generator = settings.into_generator();
    let schema = generator.into_root_schema_for::<T>();
    let mut schema_value = serde_json::to_value(&schema)?;

    let type_name: &str = std::any::type_name::<T>();
    let type_name = type_name.split("::").last().unwrap_or("ExtractorType");
    if let Some(obj) = schema_value.as_object_mut() {
        obj.remove("$schema");
        obj.remove("format");
        obj.remove("title");
    }
    process_json_value(&mut schema_value);
    let schema = json!({
        "name": type_name,
        "strict": true,
        "schema": schema_value
    });
    Ok(json!({
        "type": "json_schema",
        "json_schema": schema
    }))
}

fn process_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            let fields_to_remove = ["$schema", "format", "title", "minimum"];
            for &f in &fields_to_remove {
                if obj.get(f).map_or(false, |v| v.is_string() || v.is_number()) {
                    obj.remove(f);
                }
            }
            if let Some(v) = obj.get("oneOf").cloned() {
                obj.remove("oneOf");
                obj.insert("anyOf".to_string(), v);
            };

            if obj.contains_key("properties") {
                obj.insert("additionalProperties".to_string(), json!(false));
            }
            for (_, v) in obj.iter_mut() {
                process_json_value(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for elem in arr.iter_mut() {
                process_json_value(elem);
            }
        }
        _ => {}
    }
}

/// Serializes user messages, ignoring tool call responses
pub fn serialize_user<S>(
    content: &str,
    _tool_calls: &Option<Vec<ToolResponse>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_newtype_struct("user", &content)
}

/// Serializes assistant messages with integrated tool call information
pub fn serialize_assistant<S>(
    content: &str,
    tool_calls: &Option<Vec<ToolCall>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let combined_content = match tool_calls {
        Some(calls) => &format!("{content} {calls:?}"),
        None => content,
    };
    serializer.serialize_newtype_struct("assistant", &combined_content)
}

fn combine_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a_val), Some(b_val)) => Some(a_val + b_val),
        _ => None,
    }
}
