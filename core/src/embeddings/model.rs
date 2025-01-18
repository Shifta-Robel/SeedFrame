use async_trait::async_trait;

#[derive(Debug)]
pub enum ModelError {
    Undefined,
}

#[async_trait]
pub trait EmbeddingModel {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, ModelError>;
}
