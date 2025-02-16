use async_trait::async_trait;

use super::embeddings::embedding::Embedding;

pub mod in_memory_vec_store;
pub use in_memory_vec_store::InMemoryVectorStore;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorStoreError {
    Undefined,
    EmbeddingNotFound,
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
