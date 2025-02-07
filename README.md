# Seedframe
A simple, and robust Rust library for writing LLM applications

## Table of Contents

1.  [Features](#org3877a2e)
2.  [Installation](#org8fea25f)
3.  [Usage](#org8330441)
    1.  [Architectural overview](#orga27e92a)
    2.  [Constructing a client](#org3ee22e6)
    3.  [Using the client](#orgee09acc)
4.  [RoadMap](#org3e20cd9)



<a id="org3877a2e"></a>

## Features

-   Simple boiler-plate free setup
-   Async
-   Generic and extensible API
-   Macros


<a id="org8fea25f"></a>

## Installation

Add the library to your cargo dependencies
```toml
[dependencies]
seedframe = "0.0.1"
seedframe_macros = "0.0.1"
async-trait = "0.1.86"
tokio = { version = "1.42.0", features = ["full"] }
```

<a id="org8330441"></a>

## Usage

This library is in early stages and its API  is subject to change.


<a id="orga27e92a"></a>

### Architectural overview

Seedframe is a library for creating LLM applications like RAG systems. It provides a `Client` type that can be loaded up with data from text documents, HTTP api's or anything that can be embedded using `Embedder`s.


<a id="org3ee22e6"></a>

### Constructing a client

A client type can be defined as such:

```rust
use seedframe::{
    completion::Client, embeddings::Embedder,
    loader::builtins::file_loaders::file_once_loader::FileOnceLoader,
    providers::openai::OpenAICompletionModel,
    vector_store::in_memory_vec_store::InMemoryVectorStore,
};
use seedframe_macros::{client, embedder, loader, vector_store};

// Data can be loaded using Loaders.
// Users can write their own loaders
// or use built in loaders like `FileOnceLoader`.
#[loader(kind = "FileOnceLoader", path = "/tmp/data/fingly.txt")]
pub struct MyLoader;

// Loaded data needs to be embedded and stored
// into a vector store, like an `InMemoryVectorStore`
// for use in a client.
#[vector_store(kind = "InMemoryVectorStore")]
pub struct MyVectorStore;


// Before loaded data is stored, it needs to be embedded
// using an Embedder. This one uses the OpenAI provider
// as the client will be an OpenAI chat client.
#[embedder(kind = "OpenAIEmbeddingModel", model = "text-embedding-3-small")]
struct MyEmbedder {

    // Notice how we register our vector store, and loaders
    // here.
    //
    // We need one vector store but we can have several loaders
    #[vector_store]
    my_vector_store: MyVectorStore,

    #[loader]
    my_loader: MyLoader,
}

// Finally, we construct our client out of our embedders
#[client(kind = "OpenAICompletionModel", model = "gpt-4o-mini")]
struct MyClient {
    // we can have multiple embedders, as long as they
    // are compatible with our client.
    #[embedder]
    my_embedder: MyEmbedder,
}
```

<a id="orgee09acc"></a>

### Using the client

```rust
#[tokio::main]
async fn main() {
    // We can build our client by providing a system prompt.
    let mut c = MyClient::build(
        "you are a helpful llm that helps user with defining terms".to_string(),
    )
    .await;

    // Documents could take time to be loaded so we introduce a delay here.
    // The amount will vary based on our use case, and factors like internet
    // connection speeds.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // We then use functions provided by the `Client` type to
    // interact with our LLM.
    let resp = c.prompt("what does fingly mean").await.unwrap();

    dbg!(resp);
}
```

<a id="org3e20cd9"></a>

## RoadMap

-   [ ] More customization options
-   [ ] More providers
-   [ ] More vector store implementations
-   and more&#x2026;

