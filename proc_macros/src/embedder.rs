use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::{format_ident, quote};
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct EmbedderConfig {
    provider: syn::Type,
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
pub(crate) enum EmbedderMacroError {
    #[error(transparent)]
    ParseError(#[from] darling::Error),
    #[error("Missing field with #[vector_store] attribute")]
    MissingVectorStore,
    #[error("Unrecognized attribute {0}")]
    UnrecognizedAttribute(String),
}

pub(crate) fn embedder_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, EmbedderMacroError> {
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

    let config: EmbedderConfig = match EmbedderConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };
    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(&input, &config)?;
    Ok(quote! {
        #struct_vis struct #struct_ident{
            inner: seedframe::embeddings::Embedder,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}

fn generate_builder(
    input: &syn::ItemStruct,
    config: &EmbedderConfig,
) -> Result<proc_macro2::TokenStream, EmbedderMacroError> {
    let vector_store_type = {
        let mut vector_store = None;
        'loop_fields: for f in &input.fields {
            'loop_attrs: for a in &f.attrs {
                match &a.meta {
                    syn::Meta::Path(p) => {
                        let ident = p
                            .get_ident()
                            .ok_or(EmbedderMacroError::UnrecognizedAttribute(format!("{p:?}")))?;

                        if ident == "vector_store" {
                            vector_store = Some(f.ty.clone());
                            break 'loop_fields;
                        }
                    }
                    _ => {
                        continue 'loop_attrs;
                    }
                }
            }
        }
        vector_store.ok_or(EmbedderMacroError::MissingVectorStore)?
    };

    let loader_types = {
        let mut loaders = Vec::new();
        for f in &input.fields {
            'loop_attrs: for a in &f.attrs {
                match &a.meta {
                    syn::Meta::Path(p) => {
                        let ident = p
                            .get_ident()
                            .ok_or(EmbedderMacroError::UnrecognizedAttribute(format!("{p:?}")))?;

                        if ident == "loader" {
                            loaders.push(f.ty.clone());
                        }
                    }
                    _ => {
                        continue 'loop_attrs;
                    }
                }
            }
        }

        loaders
    };

    let vector_store_instanciated = quote! {
        ::std::sync::Arc::new(::tokio::sync::Mutex::new(::std::boxed::Box::new(#vector_store_type::build().await.unwrap().inner)))
    };
    let mut loader_instances = quote! {};

    for loader_type in loader_types {
        let string_type = loader_type.to_token_stream().to_string().to_uppercase();
        let loader_ident = format_ident!("__{}_INSTANCE", string_type);

        loader_instances.extend(quote!{
            ::std::sync::Arc::clone(&*#loader_ident,) as ::std::sync::Arc<dyn ::seedframe::loader::Loader>,
        });
    }

    let t = config.provider.clone();
    let embedding_model_init = if let Some(json_str) = &config.config {
        let json_str = serde_json::to_string(&json_str.0).unwrap();
        quote! { ::std::sync::Arc::new(::std::boxed::Box::new(#t::new(Some(#json_str)))) }
    } else {
        quote! { ::std::sync::Arc::new(::std::boxed::Box::new(#t::new(None))) }
    };

    let vis = input.clone().vis;

    Ok(quote! {
        #vis async fn build() -> Self {
            Self { inner:
                seedframe::embeddings::Embedder::init(
                    vec![#loader_instances],
                    #vector_store_instanciated,
                    #embedding_model_init,
                ).await
            }
        }
    })
}
