# Seedframe VoyageAI

Voyage AI integration crate for [Seedframe](https://github.com/Shifta-Robel/Seedframe), provides struct `VoyageAIEmbedding` which implements the trait `Seedframe::embeddings::EmbeddingModel`

Intended for use with the `#[embedder]` macro

Accepts the following configuration parameters, passed as json to the `config` attribute in the `embedder` proc-macro
    - `model`: *optional* `String` - identifier for the model to use
    - `api_key_var`: *optional* `String` - Environment variable name containing the API key
    - `api_url`: `String` - Custom API endpoint URL

Usage with the `embedder` macro:
```rust
use seedframe_voyageai::VoyageAIEmbedding;

#[embedder(
    provider = "VoyageAIEmbedding",
    config = r#"{
      "model": "voyage-3-lite",
      "api_key_var": "ENV_VAR",
      "api_url": "https://api.voyageai.com/v1/embeddings"
    }"#
)]
struct VoyageEmbedder;
```
