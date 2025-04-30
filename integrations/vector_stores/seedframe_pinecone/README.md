# seedframe pinecone

[Pinecone](https://pinecone.io) vector store integration for [Seedframe](https://github.com/Shifta-Robel/Seedframe)

Intended for use with the `vector_store` proc-macro from seedframe

Accepts the following configuration parameters, passed as json to the `config` attribute in the `vector_store` proc-macro
    - `index_host`: `String` - The host of the index to target
    - `api_key_var`: *optional* `String` - The env var to get the api key from
    - `namespace`: *optional* `String` -  The namespace of the index
    - `source_tag`: *optional* `String` - The source tag

# Examples

```rust
{
  #[vector_store(
      store = "PineconeVectorStore",
      config = r#"{
         "index_host": "https://....svc.aped.pinecone.io",
         "api_key_var": "SF_PINECONE_KEY",
         "namespace": "some_namespace",
         "source_tag": "some_tag"
      }"#
  )]
  struct Store;
}
```
