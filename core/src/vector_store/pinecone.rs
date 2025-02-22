use async_trait::async_trait;
use pinecone_sdk::{
    models::{Kind, Metadata, Namespace, QueryResponse, Value, Vector},
    pinecone::{data::Index, PineconeClientConfig},
};
use std::collections::{BTreeMap, HashMap};

use super::{VectorStore, VectorStoreError};
use crate::embeddings::embedding::Embedding;
use tokio::sync::Mutex;

struct PineconeVectorStore {
    index: Mutex<Index>,
    namespace: Namespace,
}

const PINECONE_API_VERSION: &str = "2025-01";

impl PineconeVectorStore {
    pub async fn new(
        api_key_var: Option<String>,
        index_host: String,
        source_tag: Option<String>,
        namespace: Option<String>,
    ) -> Result<Self, VectorStoreError> {
        let mut api_key = None;
        if let Some(var_name) = api_key_var {
            api_key = std::env::var(var_name).ok()
        };
        let config = PineconeClientConfig {
            api_key,
            control_plane_host: None,
            additional_headers: Some(HashMap::from([(
                "X-Pinecone-API-Version".to_string(),
                PINECONE_API_VERSION.to_string(),
            )])),
            source_tag,
        };
        let client = config.client().expect("Failed to create pinecone instance");
        let index = Mutex::new(client.index(&index_host).await?);
        let name = if let Some(n) = namespace {
            n
        } else {
            String::new()
        };
        let namespace = Namespace { name };
        Ok(Self { index, namespace })
    }
}

#[async_trait]
impl VectorStore for PineconeVectorStore {
    async fn get_by_id(&self, id: String) -> Result<Embedding, VectorStoreError> {
        let mut index_guard = self.index.lock().await;
        let resp = index_guard
            .query_by_id(&id, 1, &self.namespace, None, Some(true), Some(true))
            .await?;
        Ok(Embeddings::try_from(resp)?
            .0
            .first()
            .ok_or(VectorStoreError::EmbeddingNotFound)?
            .clone())
    }
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError> {
        let mut index_guard = self.index.lock().await;
        if embedding.raw_data.is_empty() {
            _ = index_guard
                .delete_by_id(&[&embedding.id], &self.namespace)
                .await?;
        } else {
            _ = index_guard
                .upsert(&[embedding.into()], &self.namespace)
                .await?;
        }
        Ok(())
    }
    async fn top_n(&self, query: &[f64], n: usize) -> Result<Vec<Embedding>, VectorStoreError> {
        let mut index_guard = self.index.lock().await;
        let resp = index_guard
            .query_by_value(
                query.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                None, n as u32, &self.namespace, None, Some(true), Some(true))
            .await?;
        Ok(Embeddings::try_from(resp)?.0)
    }
}

fn value_from_str(value: String) -> Value {
    let kind = Some(Kind::StringValue(value));
    Value { kind }
}

impl From<Embedding> for Vector {
    fn from(value: Embedding) -> Self {
        let id = value.id;
        let values = value.embedded_data.iter().map(|&v| v as f32).collect();
        let mut fields = BTreeMap::new();
        fields
            .insert("text".to_string(), value_from_str(value.raw_data))
            .unwrap();
        let metadata = Some(Metadata { fields });
        Vector {
            id,
            values,
            sparse_values: None,
            metadata,
        }
    }
}
struct Embeddings(Vec<Embedding>);

impl TryFrom<QueryResponse> for Embeddings {
    type Error = VectorStoreError;
    fn try_from(value: QueryResponse) -> Result<Self, Self::Error> {
        let mut embeddings: Vec<Embedding> = vec![];
        for m in value.matches {
            let mut raw_data = String::new();
            let metadata = m.metadata.ok_or(VectorStoreError::Undefined(
                "Query response without raw data".to_string(),
            ))?;
            metadata
                .fields
                .iter()
                .for_each(|(k, v)| raw_data.push_str(&format!("\"{k}\":\"{v:?}\"")));
            let embedded_data = m.values.iter().map(|&v| v as f64).collect();
            embeddings.push(Embedding {
                id: m.id,
                embedded_data,
                raw_data,
            });
        }
        Ok(Embeddings(embeddings))
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[tokio::test]
    async fn test_new_pinecone_vec_store() {
        let host = std::env::var("PINECONE_IDX_HOST").unwrap();
        let pcvs = PineconeVectorStore::new(None, host, None, None).await;
        assert!(pcvs.is_ok());
        let resp = pcvs.unwrap().index.lock().await.describe_index_stats(None).await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    async fn test_pinecone_get_by_id() {
        let host = std::env::var("PINECONE_IDX_HOST").unwrap();
        let pcvs = PineconeVectorStore::new(None, host, None, None).await;
        assert!(pcvs.is_ok());
        let resp = pcvs.unwrap().get_by_id("1".to_string()).await;
        assert!(resp.is_ok());
    }
}
