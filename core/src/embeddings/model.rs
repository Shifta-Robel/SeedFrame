use async_trait::async_trait;
use crate::embeddings::EmbedderError;

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, EmbedderError>;
}
