use seedframe::{
    completion::Client, embeddings::Embedder,
    loader::builtins::file_loaders::file_once_loader::FileOnceLoader,
    providers::openai::OpenAICompletionModel,
    vector_store::pinecone::PineconeVectorStore,
};
use seedframe_macros::{client, embedder, loader, vector_store};

#[loader(kind = "FileOnceLoader", path = "/tmp/data/**/*.txt")]
pub struct MyLoader;

#[vector_store(
    kind = "pinecone",
    host = "https://test-sf-smth.pinecone.io",
    env_var = "PINECONE_API_KEY",
    source_tag = "seedframe"
)]
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
        "Respond with the definition and language of origin for the word the user prompts you with, you'll be given a context to use for the words, if you cant get the meaning for the word from the context reply with a \"I dont know\"".to_string(),
    )
    .await;
    // delay for the vector store to finish upserting the loaded resource before the first prompt
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    _ = dbg!(c.prompt("What's a mikmak").await.unwrap());
}
