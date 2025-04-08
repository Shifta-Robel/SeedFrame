use crate::{
    completion::CompletionError, embeddings::EmbedderError, tools::ToolSetError,
    vector_store::VectorStoreError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Completion(#[from] CompletionError),
    #[error(transparent)]
    ToolSet(#[from] ToolSetError),
    #[error(transparent)]
    VectorStore(#[from] VectorStoreError),
    #[error(transparent)]
    Embedder(#[from] EmbedderError),
}
