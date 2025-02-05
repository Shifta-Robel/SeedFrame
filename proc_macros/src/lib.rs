use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};

mod loader;

#[proc_macro_error]
#[proc_macro_attribute]
pub fn loader(args: TokenStream, input: TokenStream) -> TokenStream {
    let tk_stream = loader::loader_impl(args.into(), input.into());
    if let Err(e) = tk_stream {
        abort_call_site!(e.to_string())
    }
    tk_stream.unwrap().into()
}

