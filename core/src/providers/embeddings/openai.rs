use crate::embeddings::{model::EmbeddingModel, EmbedderError};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

const DEFAULT_API_KEY_VAR_NAME: &str = "OPENAI_EMBEDDING_API_KEY";
const DEFAULT_URL: &str = "https://api.openai.com/v1/embeddings";

pub struct OpenAIEmbeddingModel {
    api_key: String,
    api_url: String,
    model: String,
    client: Client,
}

impl OpenAIEmbeddingModel {
    #[must_use] pub fn new(api_key_var: Option<String>, api_url: Option<String>, model: String) -> Self {
        let api_key_var = &api_key_var.unwrap_or(DEFAULT_API_KEY_VAR_NAME.to_string());
        let api_key = std::env::var(api_key_var).unwrap_or_else(|_| panic!("Failed to fetch env var `{api_key_var}` !"));
        let api_url = api_url.unwrap_or(DEFAULT_URL.to_string());
        Self {
            api_key,
            api_url,
            model,
            client: Client::new()
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
    async fn embed(&self, data: &str) -> Result<Vec<f64>, EmbedderError> {
        let request_body = json!({
                "input": data,
                "model": self.model,
        });
        let response = self.client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| EmbedderError::RequestError(e.to_string()))?;

        if response.status().is_success() {
            let response = response
                .json::<OpenAIEmbeddingResponse>()
                .await
                .map_err(|e| EmbedderError::ParseError(e.to_string()))?;

            Ok(response
                .data
                .into_iter()
                .flat_map(|d| d.embedding)
                .collect())
        } else {
            let error_message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            Err(EmbedderError::ProviderError(error_message))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn simple_openai_embed_request() {
        let model = "text-embedding-3-small".to_string();
        let openai_embedding_model = OpenAIEmbeddingModel::new(None, None, model);

        let response = openai_embedding_model.embed("test").await;
        assert!(response.is_ok());
    }
}
