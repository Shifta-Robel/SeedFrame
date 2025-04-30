use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use syn::{parse_macro_input, DeriveInput};

mod client;
mod embedder;
mod loader;
mod tool;
mod vector_store;

/// A proc-macro for declaring a Loader, which implements the `seedframe::loader::Loader` trait.
/// Loaders are responsible for loading resources from various sources and sending them to embedders.
///
/// # Usage with built-in Loaders
/// When using a built-in loader (like `FileOnceLoader` or `FileUpdatingLoader`), specify:
/// - `kind`: The name of the built-in loader type
/// - `path`: A glob pattern for files to load (required for file-based loaders)
///
/// ```rust,ignore
/// #[loader(
///   kind = "FileOnceLoader",
///   path = "/path/to/files/**/*.txt"
/// )]
/// pub struct MyLoader;
/// ```
///
/// # Usage with external Loaders
/// When using a custom loader implementation, specify:
/// - `external`: The type of your custom loader
/// - `config`: Optional JSON configuration for the loader
///
/// ```rust,ignore
/// #[loader(
///   external = "WebScraper",
///   config = r#"{"url": "https://example.com"}"#
/// )]
/// pub struct MyLoader;
/// ```
#[proc_macro_error]
#[proc_macro_attribute]
pub fn loader(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = loader::loader_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

/// A proc-macro for declaring a Vector Store, which stores and manages vector embeddings.
///
/// # Usage
/// Specify:
/// - `store`: The type of vector store (built-in or external)
/// - `config`: JSON configuration for the vector store
///
/// ```rust,ignore
/// #[vector_store(
///     store = "PineconeVectorStore",
///     config = r#"{"index_host": "https://example.pinecone.io"}"#
/// )]
/// pub struct MyVectorStore;
/// ```
#[proc_macro_error]
#[proc_macro_attribute]
pub fn vector_store(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = vector_store::vector_store_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

/// A proc-macro for declaring an Embedder, which converts resources into vector embeddings.
///
/// # Usage
/// Specify:
/// - `provider`: The type of embedder provider (built-in or external)
/// - `config`: JSON configuration for the embedder
///
/// ```rust,ignore
/// #[embedder(
///     provider = "VoyageAIEmbedding",
///     config = r#"{"model": "voyage-3-lite"}"#
/// )]
/// struct MyEmbedder;
/// ```
#[proc_macro_error]
#[proc_macro_attribute]
pub fn embedder(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = embedder::embedder_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

/// A proc-macro for declaring a Client, which provides completion model functionality.
///
/// # Usage
/// Required attributes:
/// - `provider`: The type of completion provider
///
/// Optional attributes:
/// - `tools`: List of tool functions to attach to the client
/// - `execution_mode`: Tool execution mode ("best_effort" or "fail_early")
/// - `config`: JSON configuration for the provider, might be an error not to specify depending on
///    the provider
///
/// ```rust,ignore
/// #[client(
///     provider = "OpenAI",
///     config = "{\"model\": \"gpt-4\"}",
///     tools("capitalize", "greet"),
///     execution_mode = "best_effort"
/// )]
/// struct MyClient;
/// ```
#[proc_macro_error]
#[proc_macro_attribute]
pub fn client(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = client::client_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

/// A proc-macro for declaring Tool functions that can be attached to Clients.
///
/// The macro parses function and argument descriptions from doc comments.
/// Documentation is required for both the function and all arguments (except State parameters).
///
/// # Usage
/// ```rust,ignore
/// /// Capitalizes all words in a string
/// /// # Arguments
/// /// * `input`: The text to capitalize
/// #[tool]
/// fn capitalize(input: String, State(state): State<AppState>) -> String {
///     input.to_uppercase()
/// }
/// ```
#[proc_macro_error]
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = tool::tool_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

#[proc_macro_derive(Extractor)]
pub fn extractor_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let expanded = quote::quote! {
        impl seedframe::completion::Extractor for #name {}
    };
    TokenStream::from(expanded)
}
