# Seedframe Anthropic

Anthropic AI integration crate for [Seedframe](https://github.com/Shifta-Robel/SeedFrame), provides struct `AnthropicClient` which implements the trait `Seedframe::completion::CompletionModel`

Intended for use with the `#[client]` macro

Accepts the following configuration parameters, passed as json to the `config` attribute in the `client` proc-macro
    - `model`: *optional* `String` - identifier for the model to use
    - `api_key_var`: *optional* `String` - Environment variable name containing the API key
    - `api_url`: *optional* `String` - Custom API endpoint URL

```rust
use seedframe_anthropic::AnthropicCompletionModel;

#[client(
    provider = "AnthropicCompletionModel",
    config = r#"{
      "model": "claude-3-7-sonnet-20250219",
      "api_key_var": "ENV_VAR",
      "api_url": "https://api.anthropic.com/v1/messages"
    }"#
)]
struct AnthropicClient;
```
