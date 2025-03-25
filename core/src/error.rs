use crate::{completion::CompletionError, embeddings::EmbedderError, tools::ToolSetError, vector_store::VectorStoreError};

#[derive(Debug)]
pub enum Error {
    Completion(CompletionError),
    ToolSet(ToolSetError),
    VectorStore(VectorStoreError),
    Embedder(EmbedderError),
}

impl From<CompletionError> for Error {
    fn from(value: CompletionError) -> Self {
        Self::Completion(value)
    }
}

impl From<ToolSetError> for Error {
    fn from(value: ToolSetError) -> Self {
        Self::ToolSet(value)
    }
}
impl From<VectorStoreError> for Error {
    fn from(value: VectorStoreError) -> Self {
        Self::VectorStore(value)
    }
}
impl From<EmbedderError> for Error {
    fn from(value: EmbedderError) -> Self {
        Self::Embedder(value)
    }
}
