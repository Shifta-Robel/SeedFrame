use async_trait::async_trait;
use uuid::Uuid;

use super::embeddings::embedding::Embedding;

#[derive(Debug)]
pub enum VectorStoreError {
    Undefined,
}

#[async_trait]
pub trait VectorStore {
    async fn get_by_id(
        &self,
        id: Uuid,
    ) -> Result<Option<Embedding>, VectorStoreError>;

    async fn store(
        &self,
        data: Embedding,
    ) -> Result<(), VectorStoreError>;

    async fn top_n(
        &self,
        query: &[f64],
        n: usize,
    ) -> Result<Vec<Embedding>, VectorStoreError>;

    async fn has(&self, id: Uuid) -> Result<bool, VectorStoreError>;
}
