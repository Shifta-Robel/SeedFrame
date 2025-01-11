use super::embeddings::embedding::Embedding;

#[derive(Debug)]
pub enum VectorStoreError {
    Undefined,
}

pub trait VectorStore {
    fn get_by_id(&self, id: usize) -> impl std::future::Future<Output = Result<Option<Embedding>, VectorStoreError>> + Send;

    fn store(
        &self,
        data: Embedding,
    ) -> impl std::future::Future<Output = Result<(), VectorStoreError>> + Send;

    fn top_n(
        self,
        query: &Embedding,
        n: usize,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, VectorStoreError>> + Send;
}
