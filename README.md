# Seedframe 🌱

A clean, macro-driven Rust library for building LLM applications.

## Features

- **Declarative API** through straight forward proc-macros
- **Modular Architecture** with clearly defined components:
  - **Loaders**: Data ingestion from various sources (files, APIs, etc.)
  - **Vector Stores**: Embedding storage and retrieval (In-memory, Redis, etc.)
  - **Embedders**: Text embedding providers
  - **LLM Clients**: Unified interface for different LLM providers
  - **Tools**: Function calling abstractions with state management and automatic documentation
  - **Extractors**: Structured output generation from LLM responses

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
seedframe = "0.1"
tokio = { version = "1.42.0", features = ["full"] }
# If you'll be using Extractors or custom types as tool-call arguments
schemars = "0.8.22"
serde = { version = "1.0.217", features = ["derive"] }
```
## Usage
This library is in early stages and its API  is subject to change.

Check out the [examples](https://github.com/Shifta-Robel/SeedFrame/tree/main/core/examples) directory for detailed usage demos.

### Tool calling and structured extraction

The `tool` proc-macro is responsible for declaring tool calls, and the `tools` attribute on the `client` proc-macro attaches them to the client.
The macro parses the descriptions for the function and it's arguments from the doc comments, its an error not to document the function or not to document every argument except the state arguments

You could also extract structured output from the llms, the target types need to implement the `Extractor`, `schemars::JsonSchema` and the `serde::Deserialize` traits.
Like the tools the description for the type and for it's fields will get extracted from the docs and get passed to the llm, but its not an error to leave them undocumented.

```rust
#[client(provider = "openai", model = "gpt-4o-mini", tools("analyze"))]
struct ToolClient;

/// Perform sentiment analysis on text
/// # Arguments
/// * `text`: Input text to analyze
/// * `language`: Language of the text (ISO 639-1)
#[tool]
fn analyze(text: String, language: String) -> String {
    todo!("implementation");
}

#[derive(Extractor)]
struct PersonData {
    /// Age in years
    age: u8,
    /// Email address
    email: String
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut client = ToolClient::build("You're a data analyst".to_string())
        .await
        .with_state(AppState::new())?;

    // Tool call
    client.prompt("Analyze this: 'I love Rust!' (en)")
        .send()
        .await?;

    // Structured extraction
    let person = client.prompt("John is 30, email john@example.com")
        .extract::<PersonData>()
        .await?;
}
```

### Building a simple RAG
```rust
use seedframe::prelude::*;

// Declare file loader that doesnt check for updates, loading files that match the glob pattern
#[loader(kind = "FileOnceLoader", path = "/tmp/data/**/*.txt")]
pub struct MyLoader;

#[vector_store(kind = "InMemoryVectorStore")]
pub struct MyVectorStore;

#[embedder(provider = "openai", model = "text-embedding-3-small")]
struct MyEmbedder {
    #[vector_store]
    my_vector_store: MyVectorStore,
    #[loader]
    my_loader: MyLoader,
}

#[client(provider = "openai", model = "gpt-4o-mini")]
struct MyClient {
    #[embedder]
    my_embedder: MyEmbedder,
}

#[tokio::main]
async fn main() {
    let mut client = MyClient::build(
        "You are a helpful assistant".to_string()
    ).await;
    
    tokio::time::sleep(Duration::from_secs(5)).await;
    let response = client.prompt("Explain quantum computing").send().await.unwrap();
}
```

## Contributing

All contributions as welcome! Writing integrations for LLM providers and Embedders is some what trivial, use the implementations for the already supported providers as inspiration.
This library could use support for more loaders, vector stores... so don't shy away from helping!


#### ⭐ Leave a Star!

If you find seedframe helpful or interesting, please consider giving it a star so more people get to see it!
