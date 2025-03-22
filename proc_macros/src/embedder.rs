use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::ToTokens;
use quote::{format_ident, quote};
use std::fmt::Display;

type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct EmbedderConfig {
    provider: String,
    #[darling(default)]
    model: Option<String>,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum EmbedderMacroError {
    UnknownEmbedderModel(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
    MissingVectorStore,
    UnrecognizedAttribute(String),
}

impl From<DarlingError> for EmbedderMacroError {
    fn from(err: DarlingError) -> Self {
        Self::ParseError(err)
    }
}

impl Display for EmbedderMacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => {
                write!(f, "Failed to parse loader macro: {e}")
            }
            Self::UnknownEmbedderModel(l) => {
                write!(
                    f,
                    "Unknown embedding model provider: '{l}'. valid options are openai,"
                )
            }
            Self::UnsupportedArgument(arg, embedder) => {
                write!(
                    f,
                    "Unsupported argument '{arg}' for '{embedder}' embedder type"
                )
            }
            Self::MissingArgument(arg, embedder) => {
                write!(
                    f,
                    "Missing required argument '{arg}' for '{embedder}' embedder type"
                )
            }
            EmbedderMacroError::MissingVectorStore => {
                write!(f, "Missing field with #[vector_store] attribute")
            }
            EmbedderMacroError::UnrecognizedAttribute(s) => write!(f, "Unrecognized attribute {s}"),
        }
    }
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
                Self::OpenAIEmbeddingModel => "OpenAIEmbeddingModel",
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

    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAIEmbeddingModel => &["model"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAIEmbeddingModel => &["model"],
        }
    }
}

fn validate_config(
    config: &EmbedderConfig,
    loader_type: &BuiltInEmbedderType,
) -> Result<(), EmbedderMacroError> {
    let required = loader_type.required_args();
    let supported = loader_type.supported_args();
    let check_arg = |name: &str, value: &Option<String>| {
        if value.is_none() && required.contains(&name) {
            Err(EmbedderMacroError::MissingArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        } else if value.is_some() && !supported.contains(&name) {
            Err(EmbedderMacroError::UnsupportedArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        } else {
            Ok(())
        }
    };
    _ = check_arg("model", &config.model)?;
    Ok(())
}
fn generate_builder(
    embedder_type: &BuiltInEmbedderType,
    vector_store_type: syn::Type,
    loader_types: Vec<syn::Type>,
    config: &EmbedderConfig,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
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

    match embedder_type {
        BuiltInEmbedderType::OpenAIEmbeddingModel => {
            let model = config.model.as_ref().unwrap().to_string();
            let embedding_model_init = quote! {
                {
                    ::std::sync::Arc::new(::std::boxed::Box::new(seedframe::providers::embeddings::openai::OpenAIEmbeddingModel::new(::std::env::var("SEEDFRAME_OPENAI_API_KEY").unwrap().to_string(), "https://api.openai.com/v1/embeddings".to_string(), #model.to_string())))
                }
            };
            quote! {
                #vis async fn build() -> Self {
                    Self { inner:
                        seedframe::embeddings::Embedder::init(
                            vec![#loader_instances],
                            #vector_store_instanciated,
                            #embedding_model_init,
                        ).await.unwrap()
                    }
                }
            }
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

    let embedder_type = BuiltInEmbedderType::from_str(&config.provider)?;
    validate_config(&config, &embedder_type)?;

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(
        &embedder_type,
        vector_store_type,
        loader_types,
        &config,
        &struct_vis,
    );

    Ok(quote! {
        #struct_vis struct #struct_ident{
            inner: seedframe::embeddings::Embedder,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}
