#[cfg(feature = "pinecone")]
use ::pinecone_sdk::utils::errors::PineconeError;
use async_trait::async_trait;
use thiserror::Error;

use super::embeddings::embedding::Embedding;

pub mod in_memory_vec_store;
#[cfg(feature = "pinecone")]
pub mod pinecone;

pub use in_memory_vec_store::InMemoryVectorStore;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum VectorStoreError {
    #[error("Failed to create store: {0}")]
    FailedToCreateStore(String),
    #[error("Failed upsert: {0}")]
    FailedUpsert(String),
    #[error("Embedding not found")]
    EmbeddingNotFound,
    #[cfg(feature = "pinecone")]
    #[error("Pinecone error: {0}")]
    Pinecone(String),
}

#[cfg(feature = "pinecone")]
impl From<PineconeError> for VectorStoreError {
    fn from(value: PineconeError) -> Self {
        Self::Pinecone(value.to_string())
    }
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Fetch an Embedding from the vec store with a matching id
    async fn get_by_id(&self, id: String) -> Result<Embedding, VectorStoreError>;

    /// Stores or updates an embedding in the vector store
    ///
    /// if the `raw_data` field is empty, it removes the embedding from the store.
    /// if the `raw_data` fiels is empty and the embedding does not exist, it returns an `EmbeddingNotFound`.
    /// if the `raw_data` is not empty, it inserts or updates the embedding in the store
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>;

    /// Fetch top n `Embedding`s ordered by cosine_similarity score
    async fn top_n(&self, query: &[f64], n: usize) -> Result<Vec<Embedding>, VectorStoreError>;
}

pub(crate) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot_product / (norm_a * norm_b)
}
