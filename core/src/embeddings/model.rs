use super::embedding::Embedding;

#[derive(Debug)]
pub enum ModelError {
    Undefined,
}

pub trait EmbeddingModel {
    fn embed(
        &self,
        data: &str,
    ) -> impl std::future::Future<Output = Result<Embedding, ModelError>> + Send;
}
