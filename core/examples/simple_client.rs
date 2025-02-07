use seedframe::{
    completion::Client, embeddings::Embedder,
    loader::builtins::file_loaders::file_once_loader::FileOnceLoader,
    providers::openai::OpenAICompletionModel,
    vector_store::in_memory_vec_store::InMemoryVectorStore,
};
use seedframe_macros::{client, embedder, loader, vector_store};

#[loader(kind = "FileOnceLoader", path = "/tmp/data/**/*.txt")]
pub struct MyLoader;

#[vector_store(kind = "InMemoryVectorStore")]
pub struct MyVectorStore;

#[embedder(kind = "OpenAIEmbeddingModel", model = "text-embedding-3-small")]
struct MyEmbedder {
    #[vector_store]
    my_vector_store: MyVectorStore,
    #[loader]
    my_loader: MyLoader,
}

#[client(kind = "OpenAICompletionModel", model = "gpt-4o-mini")]
struct MyClient {
    #[embedder]
    my_embedder: MyEmbedder,
}

#[tokio::main]
async fn main() {
    let mut c = MyClient::build(
        "You are a test llm. you will reply to the user with their own prompt".to_string(),
    )
    .await;

    let resp = c.prompt("Hey there").await.unwrap();

    dbg!(resp);
}
