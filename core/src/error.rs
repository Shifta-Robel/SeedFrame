use crate::{
    completion::CompletionError, embeddings::EmbedderError, tools::ToolSetError,
    vector_store::VectorStoreError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Completion error")]
    Completion(#[from] CompletionError),
    #[error("ToolSet error")]
    ToolSet(#[from] ToolSetError),
    #[error("VectorStore error")]
    VectorStore(#[from] VectorStoreError),
    #[error("Embedder error")]
    Embedder(#[from] EmbedderError),
}
