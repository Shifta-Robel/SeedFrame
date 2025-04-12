use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::{format_ident, quote};
use std::fmt::Display;
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct EmbedderConfig {
    #[darling(default)]
    provider: Option<String>,
    #[darling(default)]
    model: Option<String>,
    #[darling(default)]
    external: Option<syn::Type>,
    #[darling(default)]
    api_key_var: Option<String>,
    #[darling(default)]
    url: Option<String>,
}

#[derive(Debug, Error)]
pub(crate) enum EmbedderMacroError {
    #[error("Unknown embedding model provider: '{0}'. valid options are openai,")]
    UnknownEmbedderModel(String),
    #[error(transparent)]
    ParseError(#[from] darling::Error),
    #[error("Unsupported argument '{0}' for '{1}' embedder type")]
    UnsupportedArgument(String, String),
    #[error("Missing required argument '{0}' for '{1}' embedder type")]
    MissingArgument(String, String),
    #[error("Missing field with #[vector_store] attribute")]
    MissingVectorStore,
    #[error("Unrecognized attribute {0}")]
    UnrecognizedAttribute(String),
}

#[derive(Debug, Clone)]
enum BuiltInEmbedderType {
    OpenAIEmbeddingModel,
}

impl Display for BuiltInEmbedderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::OpenAIEmbeddingModel => "seedframe::providers::embeddings::openai::OpenAIEmbeddingModel",
            }
        )
    }
}

impl BuiltInEmbedderType {
    fn from_str(provider: &str) -> Result<Self, EmbedderMacroError> {
        match provider {
            "openai" => Ok(Self::OpenAIEmbeddingModel),
            unknown => Err(EmbedderMacroError::UnknownEmbedderModel(
                unknown.to_string(),
            )),
        }
    }
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

    validate_config(&config)?;
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

    let embedding_model_init = match parse_embedder_type(config)? {
        ProviderType::BuiltIn(t) => {
            let model = config.model.as_ref().unwrap().to_string();
            let api_key = if let Some(key) = &config.api_key_var {
                quote!{Some(#key.to_string())}
            }else {quote!{None}};
            let url = if let Some(url) = &config.url{
                quote!{Some(#url.to_string())}
            }else {quote!{None}};
            quote!{
                ::std::sync::Arc::new(::std::boxed::Box::new(#t::new(#api_key, #url, #model.to_string())))
            }
        },
        ProviderType::External(t) => {
            quote!{
                ::std::sync::Arc::new(::std::boxed::Box::new(#t::new()))
            }
        },
    };

    let vis = input.clone().vis;

    Ok(quote! {
        #vis async fn build() -> Self {
            Self { inner:
                seedframe::embeddings::Embedder::init(
                    vec![#loader_instances],
                    #vector_store_instanciated,
                    #embedding_model_init,
                ).await.unwrap()
            }
        }
    })
}

enum ProviderType {
    BuiltIn(syn::Type),
    External(syn::Type),
}

fn parse_embedder_type(config: &EmbedderConfig) -> Result<ProviderType, EmbedderMacroError> {
    let p_type = if config.provider.is_some() {
        let type_str = BuiltInEmbedderType::from_str(&config.provider.clone().unwrap())?.to_string();
        syn::parse_str(&type_str).map_err(|e| EmbedderMacroError::ParseError(darling::Error::from(e)))?
    }else {
        config.external.clone().unwrap()
    };

    if config.provider.is_some() {
        Ok(ProviderType::BuiltIn(p_type))
    }else {
        Ok(ProviderType::External(p_type))
    }
}

fn validate_config(config: &EmbedderConfig) -> Result<(), EmbedderMacroError> {
    if config.provider.is_some() && config.external.is_some() {
        return Err(EmbedderMacroError::ParseError(darling::Error::custom(
                    "Only one of the attributes `provider` and `external` allowed at a time!")));
    }

    if config.provider.is_none() && config.external.is_none() {
        return Err(EmbedderMacroError::ParseError(darling::Error::custom(
                    "Expected one of the attributes `provider` or `external`!")));
    }

    if config.provider.is_some() {
        ensure_required_field(&config.model, "model", "builtlin")?;
    } else {
        let unsupported_args = [
            ("api_key_var", config.api_key_var.is_some()),
            ("url", config.url.is_some()),
            ("model", config.model.is_some()),
        ];

        for (arg, is_present) in unsupported_args {
            if is_present {
                return Err(EmbedderMacroError::UnsupportedArgument(arg.to_string(), "external".to_string()));
            }
        }
    }

    Ok(())
}

fn ensure_required_field<T>(field: &Option<T>, name: &str, default: &str) -> Result<(), EmbedderMacroError> {
    if field.is_none() {
        Err(EmbedderMacroError::MissingArgument(name.to_string(), default.to_string()))
    } else {
        Ok(())
    }
}
