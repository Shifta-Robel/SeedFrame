use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use std::fmt::Display;

type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct ClientConfig {
    kind: String,
    #[darling(default)]
    model: Option<String>,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum ClientMacroError {
    UnknownCompletionModel(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
    UnrecognizedAttribute(String),
}

impl From<DarlingError> for ClientMacroError {
    fn from(err: DarlingError) -> Self {
        Self::ParseError(err)
    }
}

impl Display for ClientMacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => {
                write!(f, "Failed to parse loader macro: {e}")
            }
            Self::UnknownCompletionModel(l) => {
                write!(
                    f,
                    "Unknown embedding model kind: '{l}'. valid options are OpenAIEmbeddingModel"
                )
            }
            Self::UnsupportedArgument(arg, client) => {
                write!(
                    f,
                    "Unsupported argument '{arg}' for '{client}' client type"
                )
            }
            Self::MissingArgument(arg, client) => {
                write!(
                    f,
                    "Missing required argument '{arg}' for '{client}' client type"
                )
            }
            ClientMacroError::UnrecognizedAttribute(s) => write!(f, "Unrecognized attribute {s}"),
        }
    }
}

#[derive(Debug, Clone)]
enum BuiltInProviderType {
    OpenAICompletionModel,
}

impl Display for BuiltInProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::OpenAICompletionModel => "OpenAICompletionModel",
            }
        )
    }
}

impl BuiltInProviderType {
    fn from_str(kind: &str) -> Result<Self, ClientMacroError> {
        match kind {
            "OpenAICompletionModel" => Ok(Self::OpenAICompletionModel),
            unknown => Err(ClientMacroError::UnknownCompletionModel(
                unknown.to_string(),
            )),
        }
    }

    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAICompletionModel => &["model"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAICompletionModel => &["model"],
        }
    }
}

fn validate_config(
    config: &ClientConfig,
    provider_type: &BuiltInProviderType,
) -> Result<(), ClientMacroError> {
    let required = provider_type.required_args();
    let supported = provider_type.supported_args();
    let check_arg = |name: &str, value: &Option<String>| {
        if value.is_none() && required.contains(&name) {
            Err(ClientMacroError::MissingArgument(
                name.to_string(),
                provider_type.to_string(),
            ))
        } else if value.is_some() && !supported.contains(&name) {
            Err(ClientMacroError::UnsupportedArgument(
                name.to_string(),
                provider_type.to_string(),
            ))
        } else {
            Ok(())
        }
    };
    _ = check_arg("model", &config.model)?;
    Ok(())
}
fn generate_builder(
    provider_type: &BuiltInProviderType,
    embedder_types: Vec<syn::Type>,
    config: &ClientConfig,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
    let mut embedder_instances = quote! {};
    for embedder_type in embedder_types {
        embedder_instances.extend(quote! {
            #embedder_type::build().await.inner,
        });
    }
    match provider_type {
        BuiltInProviderType::OpenAICompletionModel => {
            let model = config.model.as_ref().unwrap().to_string();
            let completion_model_init = quote! {
                ::seedframe::providers::openai::OpenAICompletionModel::new(std::env::var("SEEDFRAME_OPENAI_API_KEY").unwrap().to_string(), "https://api.openai.com/v1/chat/completions".to_string(), #model.to_string())
            };
            quote! {
                #vis async fn build(preamble: String) -> Client<::seedframe::providers::openai::OpenAICompletionModel> {
                    Client::new(
                        #completion_model_init,
                        preamble,
                        0.5,
                        2400,
                        vec![#embedder_instances],
                    )
                }
            }
        }
    }
}

pub(crate) fn client_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, ClientMacroError> {
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

    let config: ClientConfig = match ClientConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };

    let embedder_types = {
        let mut embedders = Vec::new();
        for f in &input.fields {
            'loop_attrs: for a in &f.attrs {
                match &a.meta {
                    syn::Meta::Path(p) => {
                        let ident = p
                            .get_ident()
                            .ok_or(ClientMacroError::UnrecognizedAttribute(format!("{p:?}")))?;

                        if ident == "embedder" {
                            embedders.push(f.ty.clone());
                        }
                    }
                    _ => {
                        continue 'loop_attrs;
                    }
                }
            }
        }

        embedders
    };

    let provider_type = BuiltInProviderType::from_str(&config.kind)?;
    validate_config(&config, &provider_type)?;

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(&provider_type, embedder_types, &config, &struct_vis);
    let kind: syn::Type = syn::parse_str(&provider_type.to_string()).expect("Failed to parse type");

    Ok(quote! {
        #struct_vis struct #struct_ident{
            inner: Client<#kind>,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}
