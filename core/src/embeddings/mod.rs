pub mod embedding;
pub mod model;

use crate::loader::LoaderInstance;
use crate::vector_store::{VectorStore, VectorStoreError};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub enum EmbedderError {
    Undefined,
}

pub struct Embedder {
    loaders: Vec<LoaderInstance>,
    vector_store: Arc<Mutex<Box<dyn VectorStore>>>,
    embedding_model: Arc<Box<dyn EmbeddingModel>>,
}

impl Embedder {
    /// initialize the embedder
    pub async fn init(
        loaders: Vec<LoaderInstance>,
        vector_store: Arc<Mutex<Box<dyn VectorStore>>>,
        embedding_model: Arc<Box<dyn EmbeddingModel>>,
    ) -> Result<Self, EmbedderError> {
        let embedder = Self {
            loaders,
            vector_store,
            embedding_model,
        };
        embedder.init_loaders_listeners().await?;
        Ok(embedder)
    }

    /// initializes the listeners for the loaders
    async fn init_loaders_listeners(&self) -> Result<(), EmbedderError> {
        for loader in &self.loaders {
            let embedding_model = Arc::clone(&self.embedding_model);
            let vector_store = Arc::clone(&self.vector_store);
            let loader = Arc::clone(&*loader);

            let mut listener = loader.subscribe().await;
            tokio::spawn(async move {
                while let Ok(doc) = listener.recv().await {
                    let embedded_data = embedding_model.embed(&doc.data).await.unwrap();
                    vector_store
                        .lock()
                        .await
                        .store(Embedding {
                            id: doc.id,
                            embedded_data,
                            raw_data: doc.data,
                        })
                        .await
                        .unwrap();
                }
            });
        }
        Ok(())
    }

    /// return documents matching a query from the vector-store
    pub async fn query(
        &self,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<Embedding>, VectorStoreError> {
        let query = self.embedding_model.embed(query).await.unwrap();
        self.vector_store.lock().await.top_n(&query, top_n).await
    }
}
