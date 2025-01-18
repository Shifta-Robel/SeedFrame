pub mod embedding;
pub mod model;

use crate::loader::{Loader, LoadingStrategy};
use crate::vector_store::{VectorStore, VectorStoreError};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub enum EmbedderError {
    Undefined,
}

pub struct Embedder<V: VectorStore, M: EmbeddingModel> {
    loaders: Vec<Arc<dyn Loader>>,
    vector_store: Mutex<V>,
    embedding_model: M,
}

impl<V: VectorStore, M: EmbeddingModel> Embedder<V, M> {
    /// initialize the embedder
    pub async fn init(
        loaders: Vec<Arc<dyn Loader>>,
        vector_store: Mutex<V>,
        embedding_model: M,
    ) -> Result<Self, EmbedderError> {
        let mut embedder = Self {
            loaders,
            vector_store,
            embedding_model,
        };
        embedder.init_loaders().await?;
        Ok(embedder)
    }

    /// Initializes the loaders and stores the embedded data in the vector store.
    ///
    /// For static loaders, it loads, embeds, and stores the data. For dynamic loaders, it spawns a task to continuously
    /// process new data.
    async fn init_loaders(&mut self) -> Result<(), EmbedderError> {
        for loader in &self.loaders {
            match loader.strategy() {
                LoadingStrategy::Static => {
                    if !self.vector_store.lock().await.has(loader.id()).await.unwrap() {
                        let resource = loader.subscribe().await.recv().await.unwrap();
                        let embedded_data = self.embedding_model.embed(resource.data.as_str()).await.unwrap();
                        self.vector_store.lock().await.store(
                            Embedding { id: resource.id,embedded_data, raw_data: resource.data }
                        ).await.unwrap();
                    }
                },
                LoadingStrategy::Dynamic => {
                    if !self.vector_store.lock().await.has(loader.id()).await.unwrap() {
                        let mut listener = loader.subscribe().await;

                        while let Ok(doc) = listener.recv().await {
                            let embedded_data = self.embedding_model.embed(&doc.data).await.unwrap();
                            self.vector_store.lock().await.store(
                                Embedding { id: doc.id,embedded_data, raw_data: doc.data }
                            ).await.unwrap();
                        }
                    };
                }
            }
        }
        Ok(())
    }

    /// return documents matching a query from the vector-store
    pub async fn query(self, query: &str, top_n: usize) -> Result<Vec<Embedding>, VectorStoreError> {
        let query = self.embedding_model.embed(query).await.unwrap();
        self.vector_store.lock().await.top_n(&query, top_n).await
    }
}
