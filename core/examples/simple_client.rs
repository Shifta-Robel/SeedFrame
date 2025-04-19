use seedframe::prelude::*;
use seedframe::providers::{completions::OpenAI, embeddings::OpenAIEmbedding};
use seedframe::vector_store::InMemoryVectorStore;

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

#[client(provider = "OpenAI")]
struct MyClient {
    #[embedder]
    my_embedder: MyEmbedder,
}

#[tokio::main]
async fn main() {
    let mut c = MyClient::build(
        "Respond with the definition and language of origin for the word the user prompts you with, you'll be given a context to use for the words, if you cant get the meaning for the word from the context reply with a \"I dont know\"",
    )
    .await;
    // delay for the vector store to finish upserting the loaded resource before the first prompt
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    _ = dbg!(c.prompt("What's a mikmak").send().await.unwrap());
}
