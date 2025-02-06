use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};

mod loader;
mod vector_store;
mod embedder;
mod client;

#[proc_macro_error]
#[proc_macro_attribute]
pub fn loader(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = loader::loader_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn vector_store(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = vector_store::vector_store_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn embedder(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = embedder::embedder_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

#[proc_macro_error]
#[proc_macro_attribute]
pub fn client(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = client::client_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}
