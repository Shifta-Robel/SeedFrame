use async_trait::async_trait;
use pinecone_sdk::{
    models::{Kind, Metadata, Namespace, QueryResponse, Value, Vector},
    pinecone::{data::Index, PineconeClientConfig},
    utils::errors::PineconeError,
};
use serde::{de::Error, Serialize, Deserialize};
use std::collections::{BTreeMap, HashMap};

use seedframe::vector_store::{VectorStore, VectorStoreError};
use seedframe::embeddings::embedding::Embedding;
use tokio::sync::Mutex;

/// Configuration structure for the Pinecone client.
///
/// This is deserialized from the JSON config provided in the `#[vector_store]` macro.
///
/// # Examples
///
/// Basic configuration:
/// ```json
/// {
///     "index_host": "https://....svc.aped.pinecone.io",
///     "api_key_var": "SF_PINECONE_KEY",
///     "namespace": "some_namespace",
///     "source_tag": "some_tag"
/// }
/// ```
#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Config {
    api_key_var: Option<String>,
    index_host: String,
    source_tag: Option<String>,
    namespace: Option<String>,
}

/// A [Pinecone](https://pinecone.io) client for use with seedframe.
///
/// # Usage
///
/// Intended for use through the `#[vector_store]` proc-macro from seedframe:
/// ```ignore
/// #[vector_store(
///     store = "PineconeVectorStore",
///     config = r#"{
///       "index_host": "https://....svc.aped.pinecone.io",
///       "namespace": "some_namespace",
///     }"#
/// )]
/// struct SomeStruct;
/// ```
pub struct PineconeVectorStore {
    index: Mutex<Index>,
    namespace: Namespace,
}

const PINECONE_API_VERSION: &str = "2025-01";

impl PineconeVectorStore {
    /// Creates a new `PineconeVectorStore` from a JSON configuration string
    ///
    /// # Panics
    /// This function will panic if:
    ///  - If no json config is passed
    ///  - The provided JSON is malformed and cannot be parsed
    ///  - The JSON contains unknown fields
    ///
    /// # Errors
    /// This function will error if:
    ///  - it fails to target pinecone index
    pub async fn new(
        config_json: Option<&str>
    ) -> Result<Self, VectorStoreError> {
        assert!(config_json.is_some(), "{:?}",
                serde_json::Error::custom( "A config json with the required `index_host` expected!"));
        // if config_json.is_none() {
        //     panic!("{:?}", 
        //         serde_json::Error::custom( "A config json with the required `index_host` expected!"))
        // }
        let json_config: Config = serde_json::from_str(config_json.unwrap()).unwrap();

        let mut api_key = None;
        if let Some(var_name) = json_config.api_key_var {
            api_key = std::env::var(var_name).ok();
        };
        let config = PineconeClientConfig {
            api_key,
            control_plane_host: None,
            additional_headers: Some(HashMap::from([(
                "X-Pinecone-API-Version".to_string(),
                PINECONE_API_VERSION.to_string(),
            )])),
            source_tag: json_config.source_tag,
        };
        let client = config.client().expect("Failed to create pinecone instance");
        let index = Mutex::new(client.index(&json_config.index_host).await.map_err(into_vec_store_error)?);
        let name = json_config.namespace.unwrap_or_default();
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
            .await.map_err(into_vec_store_error)?;
        Ok(Embeddings::try_from(resp)?
            .0
            .first()
            .ok_or(VectorStoreError::EmbeddingNotFound)?
            .clone())
    }
    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError> {
        let mut index_guard = self.index.lock().await;
        if embedding.raw_data.is_empty() {
            () = index_guard
                .delete_by_id(&[&embedding.id], &self.namespace)
                .await.map_err(into_vec_store_error)?;
        } else {
            _ = index_guard
                .upsert(&[vector_from_embedding(embedding)], &self.namespace)
                .await.map_err(into_vec_store_error)?;
        }
        Ok(())
    }
    #[allow(clippy::cast_possible_truncation)]
    async fn top_n(&self, query: &[f64], n: usize) -> Result<Vec<Embedding>, VectorStoreError> {
        let mut index_guard = self.index.lock().await;
        let resp = index_guard
            .query_by_value(
                query.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                None,
                n as u32,
                &self.namespace,
                None,
                Some(true),
                Some(true),
            )
            .await.map_err(into_vec_store_error)?;
        Ok(Embeddings::try_from(resp)?.0)
    }
}

fn value_from_str(value: String) -> Value {
    let kind = Some(Kind::StringValue(value));
    Value { kind }
}

#[allow(clippy::cast_possible_truncation)]
fn vector_from_embedding(embedding: Embedding) -> Vector {
    let id = embedding.id;
    let values = embedding.embedded_data.iter().map(|&v| v as f32).collect();
    let mut fields = BTreeMap::new();
    fields.insert("text".to_string(), value_from_str(embedding.raw_data));
    let metadata = Some(Metadata { fields });
    Vector {
        id,
        values,
        sparse_values: None,
        metadata,
    }
}

struct Embeddings(Vec<Embedding>);

impl TryFrom<QueryResponse> for Embeddings {
    type Error = VectorStoreError;
    fn try_from(value: QueryResponse) -> Result<Self, Self::Error> {
        let mut embeddings: Vec<Embedding> = vec![];
        for m in value.matches {
            let mut raw_data = String::new();
            let metadata = m.metadata.ok_or(VectorStoreError::Provider(
                "Query response without raw data".to_string(),
            ))?;
            metadata
                .fields
                .iter()
                .for_each(|(k, v)| raw_data.push_str(&format!("\"{k}\":\"{v:?}\"")));
            let embedded_data = m.values.iter().map(|&v| f64::from(v)).collect();
            embeddings.push(Embedding {
                id: m.id,
                embedded_data,
                raw_data,
            });
        }
        Ok(Embeddings(embeddings))
    }
}

#[allow(clippy::needless_pass_by_value)]
fn into_vec_store_error(e: PineconeError) -> VectorStoreError {
    VectorStoreError::Provider(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_new_pinecone_vec_store() {
        let host = std::env::var("PINECONE_IDX_HOST").unwrap();
        let config = format!(r#"{{"index_host": "{}"}}"#, host);
        let pcvs = PineconeVectorStore::new(Some(&config)).await;
        assert!(pcvs.is_ok());
        let resp = pcvs
            .unwrap()
            .index
            .lock()
            .await
            .describe_index_stats(None)
            .await;
        assert!(resp.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_pinecone_get_by_id() {
        let host = std::env::var("PINECONE_IDX_HOST").unwrap();
        let config = format!(r#"{{"index_host": "{}"}}"#, host);
        let pcvs = PineconeVectorStore::new(Some(&config)).await;
        assert!(pcvs.is_ok());
        let resp = pcvs.unwrap().get_by_id("1".to_string()).await;
        assert!(resp.is_ok());
    }
}
