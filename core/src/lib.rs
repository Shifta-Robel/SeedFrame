//! # Seedframe - Core API Documentation
//!
//! Seedframe is a clean, macro-driven Rust library for building LLM applications.
//!
//! ## Features
//!
//! - **Declarative API** through straight forward proc-macros
//! - **Modular Architecture** with clearly defined components:
//!   - **Loaders**: Data ingestion from various sources (files, APIs, etc.)
//!   - **Vector Stores**: Embedding storage and retrieval (In-memory, Redis, etc.)
//!   - **Embedders**: Text embedding providers
//!   - **LLM Clients**: Unified interface for different LLM providers
//!   - **Tools**: Function calling abstractions with state management and automatic documentation
//!   - **Extractors**: Structured output generation from LLM responses
//!
//! ## Examples
//!
//! The seedframe repo contains a [number of examples](https://github.com/Shifta-Robel/SeedFrame/tree/main/core/examples) that show how to put all the pieces together.
//!
//! ### Building a simple RAG
//!
//! ```rust,no_run
//! use seedframe::prelude::*;
//! use seedframe::providers::{completions::OpenAI, embeddings::OpenAIEmbedding};
//!
//! // Declare file loader that doesnt check for updates, loading files that match the glob pattern
//! #[loader(kind = "FileOnceLoader", path = "/tmp/data/**/*.txt")]
//! pub struct MyLoader;
//!
//! #[vector_store(kind = "InMemoryVectorStore")]
//! pub struct MyVectorStore;
//!
//! #[embedder(provider = "OpenAIEmbedding")]
//! struct MyEmbedder {
//!     #[vector_store]
//!     my_vector_store: MyVectorStore,
//!     #[loader]
//!     my_loader: MyLoader,
//! }
//!
//! #[client(provider = "OpenAI")]
//! struct MyClient {
//!     #[embedder]
//!     my_embedder: MyEmbedder,
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut client = MyClient::build(
//!         "You are a helpful assistant".to_string()
//!     ).await;
//!     
//!     tokio::time::sleep(Duration::from_secs(5)).await;
//!     let response = client.prompt("Explain quantum computing").send().await.unwrap();
//! }
//! ```
//!
//! ### Tool calls and Extractors
//!
//! ```rust,no_run
//! use seedframe::{providers::completions::OpenAI, prelude::*};
//!
//! #[client(provider = "OpenAI", tools("analyze"))]
//! struct ToolClient;
//!
//! /// Perform sentiment analysis on text
//! /// # Arguments
//! /// * `text`: Input text to analyze
//! /// * `language`: Language of the text (ISO 639-1)
//! #[tool]
//! fn analyze(text: String, language: String) -> String {
//!     todo!("implementation");
//! }
//!
//! #[derive(Extractor)]
//! struct PersonData {
//!     /// Age in years
//!     age: u8,
//!     /// Email address
//!     email: String
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut client = ToolClient::build("You're a data analyst".to_string())
//!         .await
//!         .with_state(AppState::new())?;
//!
//!     // Tool call
//!     client.prompt("Analyze this: 'I love Rust!' (en)")
//!         .send()
//!         .await?;
//!
//!     // Structured extraction
//!     let person = client.prompt("John is 30, email john@example.com")
//!         .extract::<PersonData>()
//!         .await?;
//! }
//! ```
//!
//! ### Sharing state with tools
//!
//! You can pass state to tools by adding arguments of type `State<_>` to them, the only catch is that there can only be one type of State\<T\> attached to the client.
//!
//! ```rust,no_run
//! use seedframe::{providers::completions::OpenAI, prelude::*};
//!
//! #[client(provider = "OpenAI", tools("greet"))]
//! struct ToolClient;
//!
//! /// Greets a user
//! /// # Arguments
//! /// * `name`: name of the user
//! #[tool]
//! fn greet(name: String, State(count): State<u32>) -> String {
//!     for _ in range 0..count { println("Hello {name}!!")};
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut client = ToolClient::build("You're a helpful assistant".to_string())
//!         .await
//!         .with_state(3u32)?
//!         .with_state(7u32)? // this is an error since there's already a State of type u32 attached
//!         .with_state("some other state".to_string())?;
//!
//!     // Tool call
//!     client.prompt("Say hi to jack for me".to_string())
//!         .send()
//!         .await?;
//! }
//! ```
//!
//! #### Sharing mutable state with tools
//!
//! To share mutable state you can use types with interior mutablity, eg `Mutex`s
//!
//! ```rust,no_run
//! /// Greets a user
//! /// # Arguments
//! /// * `name`: name of the user
//! #[tool]
//! fn greet(name: String, State(count): State<u32>) -> String {
//!     let mut count = count.lock().unwrap();
//!     println!("{name}! This is my {count}th time saying hello");
//!     *count += 1;
//! }
//!
//! struct AppState {
//!     count: std::sync::Mutex<u32>
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut client = ToolClient::build("You're a helpful assistant".to_string())
//!         .await
//!         .with_state(AppState { count: std::sync::Mutex::new(0u32) })?;
//!
//!     // Tool call
//!     client.prompt("Say hi to jack for me".to_string())
//!         .send()
//!         .await?;
//!
//!     // Tool call
//!     client.prompt("Say hi to jack for me".to_string())
//!         .send()
//!         .await?;
//! }
//! ```
//!
//! ## Feature flags
//!
//! seedframe uses a set of [feature flags] to reduce the amount of compiled and
//! optional dependencies.
//!
//! The following optional features are available:
//!
//! Name | Description | Default?
//! ---|---|---
//! `pdf` | enables file loaders to parse PDFs | No

/// Language model completion and conversation management
///
/// Contains:
/// - Client for managing LLM interactions
/// - Message history tracking
/// - Response extraction
pub mod completion;

/// Document processing and representation utilities
///
/// Provides core types for handling text documents in embedding and retrieval workflows.
pub mod document;

/// Text embeddings support
pub mod embeddings;

/// Error types for all library operations
pub mod error;

/// Resource loading utilities
pub mod loader;

/// Convenience prelude exports
///
/// Re-exports commonly used types:
/// - macros from `seedframe_macros`
/// - `completion::State`
pub mod prelude;

/// Builtin completion and embedding model providers
pub mod providers;

/// Function calling and tool execution support
pub mod tools;

/// Vector storage and retrieval
pub mod vector_store;
