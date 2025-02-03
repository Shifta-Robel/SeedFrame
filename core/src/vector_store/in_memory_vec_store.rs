use async_trait::async_trait;
use tokio::sync::RwLock;
use std::collections::HashMap;

use crate::embeddings::embedding::Embedding;
use super::{VectorStore, VectorStoreError, cosine_similarity};

pub struct InMemoryVectorStore{
    embeddings: RwLock<HashMap<String, Embedding>>
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn get_by_id(&self, id: String) -> Result<Embedding, VectorStoreError>{
        let embeddings = self.embeddings.read().await;
        embeddings.get(&id).ok_or(VectorStoreError::EmbeddingNotFound).map(|v| v.clone())
    }

    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>{
        let mut embeddings = self.embeddings.write().await;
        if embedding.raw_data.is_empty() {
            embeddings.remove(&embedding.id).ok_or(VectorStoreError::EmbeddingNotFound)?;
        }else{
            embeddings.insert(embedding.id.clone(), embedding);
        }
        Ok(())
    }

    async fn top_n(&self, query: &[f64], n: usize) -> Result<Vec<Embedding>, VectorStoreError>{
        let embeddings = self.embeddings.read().await;
        let mut results = embeddings.clone().into_values().map(|embedding| {
            let score = cosine_similarity(query, &embedding.embedded_data);
            (score, embedding)
        }).collect::<Vec<_>>();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(n);
        Ok(results.iter().map(|(_, em)| em.clone()).collect())
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_by_id() {
        let embedding = Embedding {
            id: "id".to_string(),
            raw_data: "hello world".to_string(),
            embedded_data: vec![1.0, 2.0, 3.0],
        };
        let store = InMemoryVectorStore {
            embeddings: RwLock::new(HashMap::from([("id".to_string(), embedding.clone())])),
        };

        // test getting existing embedding
        let result = store.get_by_id("id".to_string()).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), embedding);

        // test getting non-existing embedding
        let result = store.get_by_id("non_existant_id".to_string()).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), VectorStoreError::EmbeddingNotFound);
    }

    #[tokio::test]
    async fn test_store() {
        let store = InMemoryVectorStore {
            embeddings: RwLock::new(HashMap::new()),
        };

        let embedding = Embedding {
            id: "id".to_string(),
            raw_data: "hello world".to_string(),
            embedded_data: vec![1.0, 2.0, 3.0],
        };

        // test storing new embedding
        let result = store.store(embedding.clone()).await;
        assert!(result.is_ok());

        let fetched_embedding = store.get_by_id("id".to_string()).await;
        assert!(fetched_embedding.is_ok());
        assert_eq!(fetched_embedding.unwrap(), embedding);

        // test updating existing embedding
        let updated_embedding = Embedding {
            id: "id".to_string(),
            raw_data: "shalom world".to_string(),
            embedded_data: vec![4.0, 5.0, 6.0],
        };
        let result = store.store(updated_embedding.clone()).await;
        assert!(result.is_ok());

        let fetched_embedding = store.get_by_id("id".to_string()).await;
        assert!(fetched_embedding.is_ok());
        assert_eq!(fetched_embedding.unwrap(), updated_embedding);

        // test removing embedding
        let empty_embedding = Embedding {
            id: "id".to_string(),
            raw_data: "".to_string(),
            embedded_data: vec![],
        };
        let result = store.store(empty_embedding).await;
        assert!(result.is_ok());

        let fetched_embedding = store.get_by_id("id".to_string()).await;
        assert!(fetched_embedding.is_err());
        assert_eq!(fetched_embedding.unwrap_err(), VectorStoreError::EmbeddingNotFound);
    }

    #[tokio::test]
    async fn test_top_n() {
        let (embedding1, embedding2, embedding3) = (
            Embedding {
                id: "id1".to_string(),
                raw_data: "hello world".to_string(),
                embedded_data: vec![1.0, 2.0, 3.0],
            },
            Embedding {
                id: "id2".to_string(),
                raw_data: "shalom world".to_string(),
                embedded_data: vec![4.0, 5.0, 6.0],
            },
            Embedding {
                id: "id3".to_string(),
                raw_data: "selam world".to_string(),
                embedded_data: vec![7.0, 8.0, 9.0],
            }
        );

        let store = InMemoryVectorStore {
            embeddings: RwLock::new(HashMap::from([
                ("id1".to_string(), embedding1.clone()),
                ("id2".to_string(), embedding2.clone()),
                ("id3".to_string(), embedding3.clone()),
            ])),
        };

        let query = vec![1.0, 2.0, 3.0];

        let result = store.top_n(&query, 2).await;
        assert!(result.is_ok());
        let top_n = result.unwrap();
        assert_eq!(top_n.len(), 2);
        assert_eq!(top_n[0], embedding1);
        assert_eq!(top_n[1], embedding2);
    }
}
