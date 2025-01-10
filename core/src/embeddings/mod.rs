#[allow(unused)]
pub mod embedding;
pub mod model;

use crate::loader::{direct::DirectLoader, publishing::PublishingLoader};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use crate::vector_store::VectorStore;

#[derive(Debug)]
pub enum EmbedderError {
    Undefined,
}

#[allow(dead_code)]
pub struct Embedder<V: VectorStore, M: EmbeddingModel> {
    direct_loaders: Vec<Arc<dyn DirectLoader>>,
    publishing_loaders: Vec<Arc<dyn PublishingLoader>>,

    vector_store: V,
    embedding_model: M,
}

impl<V: VectorStore, M: EmbeddingModel> Embedder<V, M> {
    /// initialize the embedder
    pub async fn init(
       direct_loaders: Vec<Arc<dyn DirectLoader>>,
       publishing_loaders: Vec<Arc<dyn PublishingLoader>>,
       vector_store: V,
       embedding_model: M,
    ) -> Result<Self, EmbedderError> {
        let mut embedder = Self {direct_loaders, publishing_loaders, vector_store, embedding_model};
        _ = embedder.embed().await;
        Ok(embedder)
    }

    /// call to embed the documents from the loaders, called when the embedder gets initialized and
    /// when new there are updates to the documents from the publishing loaders
    async fn embed(&mut self) -> Result<(), ()> {
        // for loader in &self.direct_loaders {
        //     let loaded_doc = loader.retrieve().await;
        //     if loaded_doc.embed_strategy == EmbedStrategy::Refresh {
        //         todo!("check if file changed")
        //     }
        // }
        todo!("embed the documents and update the vector-store")
    }

    /// return documents matching a query from the vector-store
    pub async fn query(self, _query: &str, _top_n: usize) -> Result<Vec<Embedding>, EmbedderError> {
        // todo!("embed the query");
        // todo!("search the vector-store with the embedded query")
        todo!("return the top-n matches")
    }
}
