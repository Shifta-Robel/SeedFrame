pub mod embedding;
pub mod model;
pub mod vector_store;

use crate::loader::{direct::DirectLoader, publishing::PublishingLoader};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use vector_store::VectorStore;

#[derive(Debug)]
pub enum EmbedderError {
    Undefined,
}


pub struct Embedder<V: VectorStore, M: EmbeddingModel> {
    direct_loaders: Vec<Arc<dyn DirectLoader>>,
    publishing_loaders: Vec<Arc<dyn PublishingLoader>>,

    vector_store: V,
    embedding_model: M,
}

impl<V: VectorStore, M: EmbeddingModel> Embedder<V, M> {
    pub async fn query(self, query: &str, top_n: usize) -> Result<Vec<Embedding>, EmbedderError> {
        let embedded_query = self.embedding_model.embed(query).await.unwrap();

        let direct_loaded_data = self.direct_loaders.iter().for_each(|loader| {
            let loaded_data = loader.retrieve().await.unwrap();
        });

        let similar_items = self
            .vector_store
            .top_n(&embedded_query, top_n)
            .await
            .unwrap();

        Ok(similar_items)
    }
}
