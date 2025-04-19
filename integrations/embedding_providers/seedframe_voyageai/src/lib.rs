use async_trait::async_trait;
use reqwest::Client;
use seedframe::embeddings::{model::EmbeddingModel, EmbedderError};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_API_KEY_VAR_NAME: &str = "VOYAGEAI_API_KEY";
const DEFAULT_URL: &str = "https://api.voyageai.com/v1/embeddings";

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    api_key_var: Option<String>,
    api_url: Option<String>,
    model: String,
}

/// Implementation of Seedframe's `EmbeddingModel` trait for [Voyage AI](https://voyageai.com).
///
/// This type is primarily designed to be used through the `#[embedder]` macro
/// rather than being instantiated directly. The macro will handle the
/// configuration parsing and model initialization.
///
/// # Supported Configuration
///
/// The model accepts the following configuration parameters:
///
/// - `model`: String identifier for the model to use
/// - `api_key_var`(optional): Environment variable name containing the API key
/// - `api_url`(optional): Custom API endpoint URL
///
/// # Examples
///
/// Usage with the `client` macro:
/// ```rust,no_run
/// use seedframe_voyageai::VoyageAIEmbedding;
///
/// #[embedder(
///     provider = "VoyageAIEmbedding",
///     config = r#"{
///       "model": "voyage-3-lite",
///       "api_key_var": "ENV_VAR",
///       "api_url": "https://api.voyageai.com/v1/embeddings"
///     }"#
/// )]
/// struct VoyageEmbedder;
/// ```
/// # Error Handling
///
/// When used with the `embedder` macro:
/// - Invalid config json will result in a compile-time error
/// - Unknown fields in configuration will be rejected
/// - Missing API keys at runtime will result in errors
pub struct VoyageAIEmbedding {
    api_key: String,
    api_url: String,
    model: String,
    client: Client,
}

impl VoyageAIEmbedding {
    /// Creates a new `VoyageEmbedder` from a JSON configuration string
    ///
    /// # Panics
    /// This function will panic if:
    ///  - The provided JSON is malformed and cannot be parsed
    ///  - The JSON contains unknown fields
    #[must_use]
    pub fn new(json_config: Option<&str>) -> Self {
        let (api_key_var, api_url, model) = if let Some(json) = json_config {
            let config: ModelConfig = serde_json::from_str(json).unwrap();
            (
                config
                    .api_key_var
                    .unwrap_or(DEFAULT_API_KEY_VAR_NAME.to_string()),
                config.api_url.unwrap_or(DEFAULT_URL.to_string()),
                config.model,
            )
        } else {
            panic!(
                "VoyageAIEmbedding expects a config json with atleast the required model field!"
            );
        };
        let api_key = std::env::var(&api_key_var)
            .unwrap_or_else(|_| panic!("Failed to fetch env var `{api_key_var}` !"));
        Self {
            api_key,
            api_url,
            client: reqwest::Client::new(),
            model,
        }
    }
}

#[derive(Deserialize)]
struct VoyageAIEmbeddingResponse {
    pub data: Vec<VoyageAIEmbeddingData>,
}

#[derive(Deserialize)]
struct VoyageAIEmbeddingData {
    pub embedding: Vec<f64>,
}

#[async_trait]
impl EmbeddingModel for VoyageAIEmbedding {
    async fn embed(&self, data: &str) -> Result<Vec<f64>, EmbedderError> {
        let request_body = json!({
                "input": data,
                "model": self.model,
        });
        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| EmbedderError::RequestError(e.to_string()))?;

        if response.status().is_success() {
            let response = response
                .json::<VoyageAIEmbeddingResponse>()
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
        let model = "voyage-large-2".to_string();
        let openai_embedding_model = VoyageAIEmbedding::new(None);

        let response = openai_embedding_model.embed("test").await;
        assert!(response.is_ok());
    }
}
