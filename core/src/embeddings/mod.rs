pub mod embedding;
pub mod model;
use crate::{
    loader::LoaderInstance,
    vector_store::{VectorStore, VectorStoreError},
};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("Request error: {0}")]
    RequestError(String),
    #[error("Response parse error: {0}")]
    ParseError(String),
    #[error("Provider error: {0}")]
    ProviderError(String),
}

/// The `Embedder` listens to loaders, generates embeddings for incoming documents,
/// and stores them in a vector store. It also provides functionality to query the vector store
pub struct Embedder {
    /// A list of loaders to listen to for new documents.
    loaders: Vec<LoaderInstance>,
    /// Vector store for storing and querying embeddings.
    vector_store: Arc<Mutex<Box<dyn VectorStore>>>,
    /// An embedding model used to generate embeddings from raw data.
    embedding_model: Arc<Box<dyn EmbeddingModel>>,
}

impl Embedder {
    /// Initializes the `Embedder` with the provided loaders, vector store, and embedding model.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `Embedder` instance.
    /// * `Err(EmbedderError)` - An error if initialization fails.
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

    /// Initializes listeners for the loaders.
    ///
    /// This method spawns asynchronous tasks to listen for new documents from the loaders,
    /// generate embeddings, and store them in the vector store.
    ///
    /// # Returns
    /// * `Ok(())` - If the listeners are successfully initialized.
    /// * `Err(EmbedderError)` - If an error occurs during initialization.
    async fn init_loaders_listeners(&self) -> Result<(), EmbedderError> {
        for loader in &self.loaders {
            let embedding_model = Arc::clone(&self.embedding_model);
            let vector_store = Arc::clone(&self.vector_store);
            let loader = Arc::clone(loader);

            let mut listener = loader.subscribe().await;
            tokio::spawn(async move {
                while let Ok(doc) = listener.recv().await {
                    let embedded_data = if !&doc.data.is_empty() {
                        embedding_model.embed(&doc.data).await.unwrap()
                    } else {
                        vec![]
                    };
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

    /// Queries the vector store for documents similar to the provided query.
    ///
    /// # Arguments
    /// * `query` - The query string to search for.
    /// * `top_n` - The number of top results to return.
    ///
    /// # Returns
    /// * `Ok(Vec<Embedding>)` - A list of the top `n` embeddings matching the query.
    /// * `Err(VectorStoreError)` - An error if the query fails.
    pub async fn query(
        &self,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<Embedding>, VectorStoreError> {
        let query = self.embedding_model.embed(query).await.unwrap();
        self.vector_store.lock().await.top_n(&query, top_n).await
    }
}
