pub mod embedding;
pub mod model;
use crate::{loader::LoaderInstance, vector_store::VectorStore};
use embedding::Embedding;
use model::EmbeddingModel;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{error, info};

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
    pub async fn init(
        loaders: Vec<LoaderInstance>,
        vector_store: Arc<Mutex<Box<dyn VectorStore>>>,
        embedding_model: Arc<Box<dyn EmbeddingModel>>,
    ) -> Self {
        let embedder = Self {
            loaders,
            vector_store,
            embedding_model,
        };
        embedder.init_loaders_listeners().await;
        embedder
    }

    /// Initializes listeners for the loaders.
    ///
    /// This method spawns asynchronous tasks to listen for new documents from the loaders,
    /// generate embeddings, and store them in the vector store.
    async fn init_loaders_listeners(&self) {
        for loader in &self.loaders {
            info!("Initializing loader");
            let embedding_model = Arc::clone(&self.embedding_model);
            let vector_store = Arc::clone(&self.vector_store);
            let loader = Arc::clone(loader);

            let mut listener = loader.subscribe().await;
            tokio::spawn(async move {
                info!("Spawned a thread for loader");
                while let Ok(doc) = listener.recv().await {
                    info!("Recieved document :{}", &doc.id);
                    let embedded_data = if doc.data.is_empty() {
                        vec![]
                    } else {
                        embedding_model.embed(&doc.data).await.unwrap()
                    };
                    match vector_store
                        .lock()
                        .await
                        .store(Embedding {
                            id: doc.id.clone(),
                            embedded_data,
                            raw_data: doc.data,
                        })
                        .await
                    {
                        Ok(()) => {
                            info!(
                                "Added embedding for document {} to the vector store",
                                &doc.id
                            );
                        }
                        Err(e) => {
                            error!(error = ?e, "Failed to store embedding for document {}", &doc.id);
                            panic!("{e}");
                        }
                    };
                }
            });
        }
    }

    /// Queries the vector store for documents similar to the provided query.
    ///
    /// # Arguments
    /// * `query` - The query string to search for.
    /// * `top_n` - The number of top results to return.
    ///
    /// # Returns
    /// * - A list of the top `n` embeddings matching the query.
    ///
    /// # Errors
    ///  returns `Err(seedframe::error::Error)` - If embedding the query or fetching from vec store fails.
    pub async fn query(
        &self,
        query: &str,
        top_n: usize,
    ) -> Result<Vec<Embedding>, crate::error::Error> {
        let query = self.embedding_model.embed(query).await?;
        self.vector_store
            .lock()
            .await
            .top_n(&query, top_n)
            .await
            .map_err(Into::into)
    }
}
