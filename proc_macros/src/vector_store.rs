use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct VectorStoreConfig {
    #[darling(default)]
    store: Option<syn::Type>,
    #[darling(default)]
    config: Option<JsonStr>,
}

#[derive(Debug, Clone)]
struct JsonStr(serde_json::Value);
impl FromMeta for JsonStr {
    fn from_string(value: &str) -> darling::Result<Self> {
        let value: serde_json::Value = serde_json::from_str(value)
            .map_err(|e| darling::Error::custom(format!("Invalid JSON: {e}")))?;

        Ok(JsonStr(value))
    }
}
#[derive(Debug, Error)]
pub(crate) enum VectorStoreMacroError {
    #[error(transparent)]
    ParseError(#[from] darling::Error),
}

pub(crate) fn vector_store_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, VectorStoreMacroError> {
    let attr_args: Vec<NestedMeta> = match NestedMeta::parse_meta_list(args) {
        Ok(v) => v,
        Err(e) => {
            return Err(darling::Error::from(e).into());
        }
    };
    let input: syn::ItemStruct = match syn::parse2::<syn::ItemStruct>(input) {
        Ok(is) => is,
        Err(e) => {
            return Err(darling::Error::from(e).into());
        }
    };

    let config: VectorStoreConfig = match VectorStoreConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let kind = config.store.clone().unwrap();
    let builder_impl = generate_builder(&config, &kind, struct_vis);

    Ok(quote! {
        #struct_vis struct #struct_ident{
            pub inner: #kind,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}

fn generate_builder(
    config: &VectorStoreConfig,
    kind: &syn::Type,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
    let init_store = if let Some(config) = &config.config {
        let config = serde_json::to_string(&config.0).unwrap();
        quote! { #kind::new(Some(#config)).await.unwrap() }
    } else {
        quote! { #kind::new(None).await.unwrap() }
    };
    quote! {
        #vis async fn build() -> Result<Self, seedframe::vector_store::VectorStoreError> {
            Ok(Self {
                inner: #init_store
            })
        }
    }
}
