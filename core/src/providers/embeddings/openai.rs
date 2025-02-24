use crate::embeddings::model::{EmbeddingModel, ModelError};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

pub struct OpenAIEmbeddingModel {
    api_url: String,
    api_key: String,
    model: String,
}

impl OpenAIEmbeddingModel {
    pub fn new(api_key: String, api_url: String, model: String) -> Self {
        Self {
            api_url,
            api_key,
            model,
        }
    }
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    pub data: Vec<OpenAIEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    pub embedding: Vec<f64>,
}

#[async_trait]
impl EmbeddingModel for OpenAIEmbeddingModel {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, ModelError> {
        let client = Client::new();
        let request_body = json!({
                "input": data,
                "model": self.model,
        });
        let response = client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ModelError::RequestError(e.to_string()))?;

        if response.status().is_success() {
            let response = response
                .json::<OpenAIEmbeddingResponse>()
                .await
                .map_err(|e| ModelError::ParseError(e.to_string()))?;

            Ok(response
                .data
                .into_iter()
                .map(|d| d.embedding)
                .flatten()
                .collect())
        } else {
            let error_message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(ModelError::ProviderError(error_message))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simple_openai_embed_request() {
        let api_key = std::env::var("SEEDFRAME_TEST_OPENAI_KEY")
            .unwrap()
            .to_string();
        let api_url = "https://api.openai.com/v1/embeddings".to_string();
        let model = "text-embedding-3-small".to_string();

        let openai_embedding_model = OpenAIEmbeddingModel::new(api_key, api_url, model);

        let response = openai_embedding_model.embed("test").await;

        assert!(response.is_ok());
    }
}
