#[allow(unused)]
pub mod embedding;
pub mod model;

use crate::loader::{direct::DirectLoader, publishing::PublishingLoader};
use embedding::Embedding;
use model::{EmbeddingModel, ModelError};
use std::sync::Arc;
use crate::vector_store::VectorStore;

#[derive(Debug)]
pub enum EmbedderError {
    Undefined,
}

impl From<ModelError> for EmbedderError {
    fn from(value: ModelError) -> Self {
        match value {
            ModelError::Undefined => Self::Undefined
        }
    }
}

/// Defines strategy to be used when changes to a loaded resources are detected
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EmbeddingUpdateStrategy {
    /// Append the new embedding as a new entry
    AppendAsNew,
    /// Overwrite the Embedding with the new embedded data
    OverwriteExisting,
    /// Retain only the initial version of resources ignoring changes
    IgnoreUpdates
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
    async fn embed(&mut self) -> Result<(), EmbedderError> {
        for loader in &self.direct_loaders {
            let loaded_doc = loader.retrieve().await.unwrap();
            let vec_store_fetch_result = self.vector_store.get_by_id(loaded_doc.document.id).await.unwrap();
            if let Some(_embedding) = vec_store_fetch_result {
                match loaded_doc.strategy {
                    EmbeddingUpdateStrategy::IgnoreUpdates => {},
                    EmbeddingUpdateStrategy::AppendAsNew => {
                        // check if data changed
                        let embedded_data = self.embedding_model.embed(loaded_doc.document.data.as_str()).await.unwrap();
                        let new_id = loaded_doc.document.id + 1;
                        self.vector_store.store(Embedding {
                            id: new_id,
                            raw_data: loaded_doc.document.data,
                            embedded_data
                        }).await.unwrap();
                    },
                    EmbeddingUpdateStrategy::OverwriteExisting => {
                        // check if data changed
                        let embedded_data = self.embedding_model.embed(loaded_doc.document.data.as_str()).await.unwrap();
                        self.vector_store.store(Embedding {
                            id: loaded_doc.document.id,
                            raw_data: loaded_doc.document.data,
                            embedded_data
                        }).await.unwrap();
                    }
                }
            }else{
                let embedded_data = self.embedding_model.embed(loaded_doc.document.data.as_str()).await.unwrap();
                self.vector_store.store(Embedding {
                    id: loaded_doc.document.id,
                    raw_data: loaded_doc.document.data,
                    embedded_data
                }).await.unwrap();
            }
        }
        Ok(())
    }

    /// return documents matching a query from the vector-store
    pub async fn query(self, _query: &str, _top_n: usize) -> Result<Vec<Embedding>, EmbedderError> {
        // todo!("embed the query");
        // todo!("search the vector-store with the embedded query")
        todo!("return the top-n matches")
    }
}
