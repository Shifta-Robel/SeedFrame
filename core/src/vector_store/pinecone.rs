use async_trait::async_trait;
use reqwest::{Client, Url, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};

use crate::embeddings::embedding::Embedding;
use super::{VectorStore, VectorStoreError};

const RETRIES: u8 = 3;

#[derive(Debug)]
pub struct PineConeConfig {
    api_key: String,
    host: String,
    namespace: Option<String>
}

#[derive(Debug)]
pub struct PineConeClient{
    client: Client,
    config: PineConeConfig,
    base_url: Url,
}

impl PineConeClient {
    pub async fn new(config: PineConeConfig) -> Result<Self, VectorStoreError> {
        let client = Client::new();
        let base_url = Url::parse(&format!("https://{}",config.host)).map_err(|e| VectorStoreError::FailedToCreateStore(e.to_string()))?;
        let mut attempts = 0u8;
        loop {
            let response = client
                .get(base_url.join("describe_index_stats").unwrap())
                .header("Api-Key", &config.api_key)
                .send()
                .await
                .map_err(|e| { VectorStoreError::Undefined(e.to_string()) })?;

            if response.status().is_success() {break} else{
                attempts += 1;
                if attempts >= RETRIES {
                    Err(VectorStoreError::FailedToCreateStore(
                        format!("Failed to fetch index: {:?}", response.error_for_status().unwrap_err()),
                    ))?;
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }

        Ok(Self {
            client,
            config,
            base_url,
        })
    }
}

#[async_trait]
impl VectorStore for PineConeClient {
    async fn get_by_id(&self, id: String) -> Result<Embedding, VectorStoreError>{
        use VectorStoreError::Undefined;
        let mut url = self.base_url.join("vectors/fetch").unwrap();
        if let Some(namespace) = &self.config.namespace {
            url.set_query(Some(&format!("namespace={namespace}")))
        };

        let response = self.client
            .get(url.clone())
            .header("Api-Key", &self.config.api_key)
            .query(&[("ids", &id)])
            .send()
            .await
            .map_err(|e| Undefined(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                let res: PineconeFetchResponse = response.json().await.map_err(|e| Undefined(e.to_string()))?;
                res.vectors
                    .into_iter()
                    .next()
                    .and_then(|(_, v)| v.try_into().ok())
                    .ok_or(VectorStoreError::EmbeddingNotFound)
            }
            StatusCode::NOT_FOUND => Err(VectorStoreError::EmbeddingNotFound),
            _ => Err(Undefined(response.error_for_status().unwrap_err().to_string())),
        }
    }

    async fn store(&self, embedding: Embedding) -> Result<(), VectorStoreError>{
        if embedding.raw_data.is_empty() {
            let url = self.base_url.join("vectors/delete").unwrap();
            let response = self.client
                .post(url)
                .header("Api-Key", &self.config.api_key)
                .json(&json!({ "ids": [embedding.id] }))
                .send()
                .await
                .map_err(|e| VectorStoreError::Undefined(e.to_string()))?;

            match response.status() {
                StatusCode::OK => Ok(()),
                StatusCode::NOT_FOUND => Err(VectorStoreError::EmbeddingNotFound),
                _ => Err(VectorStoreError::FailedUpsert(response.error_for_status().unwrap_err().to_string())),
            }?;
            Ok(())
        } else {
            let url = self.base_url.join("vectors/upsert").unwrap();
            let vector: Vec<PineconeRecord> = vec![embedding.try_into()?];
            
            let response = self.client
                .post(url)
                .header("Api-Key", &self.config.api_key)
                .json(&json!({ "vectors": vector }))
                .send()
                .await
                .map_err(|e| VectorStoreError::Undefined(e.to_string()))?;

            if response.status().is_success() {
                Ok(())
            } else {
                Err(VectorStoreError::FailedUpsert(response.error_for_status().unwrap_err().to_string()))
            }
        }
    }

    async fn top_n(&self, query: &[f64], n: usize) -> Result<Vec<Embedding>, VectorStoreError>{
        let url = self.base_url.join("query").unwrap();
        let response = self.client
            .post(url)
            .header("Api-Key", &self.config.api_key)
            .json(&json!({
                "vector": query,
                "topK": n,
                "includeValues": true,
                "includeMetadata": true
            }))
            .send()
            .await
            .map_err(|e| VectorStoreError::Undefined(e.to_string()))?;

        if !response.status().is_success() {
            return Err(VectorStoreError::Undefined(response.error_for_status().unwrap_err().to_string()));
        }

        let res: PineconeQueryResponse = response.json().await.map_err(|e| VectorStoreError::Undefined(e.to_string()))?;
        res.matches
            .into_iter()
            .map(|m| m.try_into())
            .collect()
    }
}

// Pinecone API data structures
#[derive(Debug, Serialize, Deserialize, Clone)]
struct PineconeRecord {
    id: String,
    values: Vec<f64>,
    metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PineconeVector {
    vectors: Vec<PineconeRecord>
}

#[derive(Debug, Serialize, Deserialize)]
struct PineconeMetadata {
    meta: BTreeMap<String, serde_json::Value>,
    #[serde(skip)]
    raw_data: String,
}

#[derive(Debug, Deserialize)]
struct PineconeFetchResponse {
    vectors: HashMap<String, PineconeRecord>
}

#[derive(Debug, Deserialize)]
struct PineconeQueryResponse {
    matches: Vec<PineconeMatch>,
}

#[derive(Debug, Deserialize)]
struct PineconeMatch {
    id: String,
    values: Vec<f64>,
    metadata: PineconeMetadata,
    // score: f64,
}

impl TryFrom<Embedding> for PineconeRecord {
    type Error = VectorStoreError;

    fn try_from(embedding: Embedding) -> Result<Self, Self::Error> {
        let mut metadata =  BTreeMap::new();
        metadata.insert("text".to_string(), serde_json::Value::from(embedding.raw_data.clone()));
        Ok(Self {
            id: embedding.id,
            values: embedding.embedded_data,
            metadata,
        })
    }
}

impl TryFrom<PineconeRecord> for Embedding {
    type Error = VectorStoreError;

    fn try_from(record: PineconeRecord) -> Result<Self, Self::Error> {
        let mut raw_data = String::new();
        record.metadata.iter().for_each(|(k, v)| raw_data.push_str(&format!("\"{k}\":\"{v}\"")));
        Ok(Self {
            id: record.id,
            embedded_data: record.values,
            raw_data,
        })
    }
}

impl TryFrom<PineconeMatch> for Embedding {
    type Error = VectorStoreError;

    fn try_from(pc_match: PineconeMatch) -> Result<Self, Self::Error> {
        Ok(Self {
            id: pc_match.id,
            embedded_data: pc_match.values,
            raw_data: pc_match.metadata.raw_data,
        })
    }
}
