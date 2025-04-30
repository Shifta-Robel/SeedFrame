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
tokio = "1.44"
async-trait = "0.1"
# If you'll be using Extractors or custom types as tool-call arguments
schemars = "0.8.22"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
dashmap = "6.1"
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
use seedframe::prelude::*;
use seedframe::providers::completions::OpenAI;

#[client(provider = "OpenAI", tools("analyze"))]
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
use seedframe::providers::{completions::OpenAI, embeddings::OpenAIEmbedding};
use seedframe::vector_store::InMemoryVectorStore;

// Declare file loader that doesnt check for updates, loading files that match the glob pattern
#[loader(kind = "FileOnceLoader", path = "/tmp/data/**/*.txt")]
pub struct MyLoader;

#[vector_store(store = "InMemoryVectorStore")]
pub struct MyVectorStore;

#[embedder(provider = "OpenAIEmbedding")]
struct MyEmbedder {
    #[vector_store]
    my_vector_store: MyVectorStore,
    #[loader]
    my_loader: MyLoader,
}

#[client(provider = "OpenAI", config = r#"{"model": "gpt-4o-mini"}"#)]
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
## Built-in Components

**Loaders**
- [`FileOnceLoader`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/loader/builtins/file_loaders/file_once_loader.rs) - Load files once using glob patterns
- [`FileUpdatingLoader`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/loader/builtins/file_loaders/file_updating_loader.rs)  - Load files and watch for changes

**Vector Stores**
- [`InMemoryVectorStore`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/vector_store/in_memory_vec_store.rs)  - Simple in-memory vector storage implementation

**Completion Providers**
- [`OpenAI`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/providers/completions/openai.rs) - [OpenAI](https://openai.com) API integration
- [`Deepseek`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/providers/completions/deepseek.rs)  - [Deepseek](https://deepseek.com) API integration
- [`Xai`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/providers/completions/xai.rs)  - [Xai](https://x.ai)'s API integration

**Embeddings**
- [`OpenAI`](https://github.com/Shifta-Robel/SeedFrame/blob/main/core/src/providers/embeddings/openai.rs) - [OpenAI](https://openai.com) embeddings API integration

---

## Integrations

SeedFrame supports extending functionality through external crates. To create an integration all thats needed is to provide a type that implements the relevant trait from seedframe (`Loader`, `CompletionModel`, etc.). You can use the following crates as inspiration if you want to write an integration crate of your own.

**Completion Providers**
- [`seedframe_anthropic`](https://github.com/Shifta-Robel/SeedFrame/tree/main/integrations/completion_providers/seedframe_anthropic)  - [Anthropic](https://anthropic.com) API integration

**Embedding Providers**
- [`seedframe_voyageai`](https://github.com/Shifta-Robel/SeedFrame/tree/main/integrations/embedding_providers/seedframe_voyageai)  - [VoyageAI](https://voyageai.com) embeddings

**Vector Stores**
- [`seedframe_pinecone`](https://github.com/Shifta-Robel/SeedFrame/tree/main/integrations/vector_stores/seedframe_pinecone)  - [Pinecone](https://pinecone.io) vector database integration

**Loaders**
- [`seedframe_webscraper`](https://github.com/Shifta-Robel/SeedFrame/tree/main/integrations/seedframe_webscraper)  - Web scraping using [scraper-rs](https://docs.rs/scraper)

---

If you wrote an integration crate, please update this list and [submit a PR](https://github.com/Shifta-Robel/SeedFrame/compare).
## Contributing

All contributions as welcome! This library could use support for more loaders, vector stores, providers ... so don't shy away from helping!


#### ⭐ Leave a Star!

If you find seedframe helpful or interesting, please consider giving it a star so more people get to see it!
