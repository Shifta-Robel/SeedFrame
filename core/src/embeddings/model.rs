use async_trait::async_trait;

#[derive(Debug)]
pub enum ModelError {
    RequestError(String),
    ParseError(String),
    ProviderError(String),
}

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, ModelError>;
}
