use crate::embeddings::{model::EmbeddingModel, EmbedderError};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, error, info, instrument};

const DEFAULT_API_KEY_VAR_NAME: &str = "OPENAI_EMBEDDING_API_KEY";
const DEFAULT_URL: &str = "https://api.openai.com/v1/embeddings";
const DEFAULT_MODEL: &str = "text-embedding-3-small";

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    api_key: Option<String>,
    api_url: Option<String>,
    model: Option<String>,
}

pub struct OpenAIEmbedding {
    api_key: String,
    api_url: String,
    model: String,
    client: Client,
}

impl OpenAIEmbedding {
    #[instrument]
    #[must_use]
    pub fn new(json_config: Option<&str>) -> Self {
        let (api_key_var, api_url, model) = if let Some(json) = json_config {
            let config = match serde_json::from_str::<ModelConfig>(json) {
                Ok(config) => config,
                Err(e) => {
                    let e = format!("Failed to deserialize json config: {e}");
                    error!(e);
                    panic!("{e}");
                }
            };
            (
                config
                    .api_key
                    .unwrap_or(DEFAULT_API_KEY_VAR_NAME.to_string()),
                config.api_url.unwrap_or(DEFAULT_URL.to_string()),
                config.model.unwrap_or(DEFAULT_MODEL.to_string()),
            )
        } else {
            (
                DEFAULT_API_KEY_VAR_NAME.to_string(),
                DEFAULT_URL.to_string(),
                DEFAULT_MODEL.to_string(),
            )
        };
        let api_key = match std::env::var(&api_key_var) {
            Ok(key) => key,
            Err(e) => {
                let e = format!("Failed to fetch env var `{api_key_var}`!, {e}");
                error!(e);
                panic!("{e}");
            }
        };
        Self {
            api_key,
            api_url,
            client: reqwest::Client::new(),
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
impl EmbeddingModel for OpenAIEmbedding {
    #[instrument(
        skip(self, data),
        fields(
            model = self.model,
            api_url = self.api_url,
            input_length = data.len()
        )
    )]
    async fn embed(&self, data: &str) -> Result<Vec<f64>, EmbedderError> {
        info!("Preparing embedding request");
        let request_body = json!({
                "input": data,
                "model": self.model,
        });
        debug!(
            request_body.input_length = data.len(),
            "Sending embedding request"
        );
        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                error!(error = ?e, "Embedding request failed");
                EmbedderError::RequestError(e.to_string())
            })?;

        let status = response.status();
        debug!(%status, "Received embedding response");

        if status.is_success() {
            let response = response
                .json::<OpenAIEmbeddingResponse>()
                .await
                .map_err(|e| {
                    error!(error = ?e, "Failed to parse embedding response");
                    EmbedderError::ParseError(e.to_string())
                })?;

            let embeddings: Vec<f64> = response
                .data
                .into_iter()
                .flat_map(|d| d.embedding)
                .collect();
            info!(
                embedding_length = embeddings.len(),
                "Successfully generated embeddings"
            );
            Ok(embeddings)
        } else {
            let error_message = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            error!(
                status = %status,
                error = %error_message,
                "Embedding API returned error"
            );

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
        let openai_embedding_model = OpenAIEmbedding::new(None);

        let response = openai_embedding_model.embed("test").await;
        assert!(response.is_ok());
    }
}
