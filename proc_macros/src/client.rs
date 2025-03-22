use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use std::fmt::Display;
use syn::{parse::Parser, Meta};

type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct ClientConfig {
    provider: String,
    #[darling(default)]
    model: Option<String>,
    #[darling(default)]
    tools: Option<ToolNames>,
    #[darling(default)]
    execution_mode: Option<String>,
}

#[derive(Clone, Debug)]
struct ToolNames(Vec<String>);

impl FromMeta for ToolNames {
    fn from_meta(meta: &Meta) -> darling::Result<Self> {
        let mut list = Vec::new();
        match meta {
            Meta::List(meta_list) => {
                let parser =
                    syn::punctuated::Punctuated::<syn::LitStr, syn::Token![,]>::parse_terminated;
                let literals = parser
                    .parse(meta_list.tokens.clone().into())
                    .map_err(|e| darling::Error::from(e))?;
                for lit in literals {
                    list.push(lit.value());
                }
            }
            _ => return Err(darling::Error::unexpected_type("expected list").with_span(meta)),
        }

        Ok(ToolNames(list))
    }
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum ClientMacroError {
    UnknownCompletionModel(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
    UnrecognizedAttribute(String),
    UnknownExecutionMode(String),
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
                write!(f, "Failed to parse client macro: {e}")
            }
            Self::UnknownCompletionModel(l) => {
                write!(
                    f,
                    "Unknown completion model provider: '{l}'. valid options are openai, deepseek, xai"
                )
            }
            Self::UnknownExecutionMode(l) => {
                write!(
                    f,
                    "Unknown execution mode : '{l}'. valid options are fail_early or best_effort"
                )
            }
            Self::UnsupportedArgument(arg, client) => {
                write!(f, "Unsupported argument '{arg}' for '{client}' client type")
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
    DeepseekCompletionModel,
    XaiCompletionModel,
}

impl Display for BuiltInProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::OpenAICompletionModel =>
                    "seedframe::providers::completions::openai::OpenAICompletionModel",
                Self::DeepseekCompletionModel =>
                    "seedframe::providers::completions::deepseek::DeepseekCompletionModel",
                Self::XaiCompletionModel =>
                    "seedframe::providers::completions::xai::XaiCompletionModel",
            }
        )
    }
}

impl BuiltInProviderType {
    fn from_str(provider: &str) -> Result<Self, ClientMacroError> {
        match provider {
            "openai" => Ok(Self::OpenAICompletionModel),
            "deepseek" => Ok(Self::DeepseekCompletionModel),
            "xai" => Ok(Self::XaiCompletionModel),
            unknown => Err(ClientMacroError::UnknownCompletionModel(
                unknown.to_string(),
            )),
        }
    }

    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAICompletionModel => &["model"],
            Self::DeepseekCompletionModel => &["model"],
            Self::XaiCompletionModel => &["model"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAICompletionModel => &["model"],
            Self::DeepseekCompletionModel => &["model"],
            Self::XaiCompletionModel => &["model"],
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
enum ExecutionModeType {
    FailEarly,
    BestEffort,
}

impl Display for ExecutionModeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::FailEarly => "seedframe::tools::ExecutionStrategy::FailEarly",
                Self::BestEffort => "seedframe::tools::ExecutionStrategy::BestEffort",
            }
        )
    }
}

impl ExecutionModeType {
    fn from_str(provider: &str) -> Result<Self, ClientMacroError> {
        match provider {
            "best_effort" => Ok(Self::BestEffort),
            "fail_early" => Ok(Self::FailEarly),
            unknown => Err(ClientMacroError::UnknownExecutionMode(unknown.to_string())),
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
    check_arg("model", &config.model)?;
    Ok(())
}
fn generate_builder(
    provider_type: &BuiltInProviderType,
    embedder_types: &[syn::Type],
    config: &ClientConfig,
    vis: &syn::Visibility,
    tool_set: &TokenStream,
) -> TokenStream {
    let mut embedder_instances = quote! {};
    for embedder_type in embedder_types {
        embedder_instances.extend(quote! {
            #embedder_type::build().await.inner,
        });
    }
    match provider_type {
        BuiltInProviderType::OpenAICompletionModel => {
            let model = config.model.as_ref().unwrap().to_string();
            let api_key = "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".to_string();
            let completion_model_init = quote! {
                ::seedframe::providers::completions::openai::OpenAICompletionModel::new(#api_key.to_string(), "https://api.openai.com/v1/chat/completions".to_string(), #model.to_string())
            };
            quote! {
                #vis async fn build(preamble: String) -> seedframe::completion::Client<::seedframe::providers::completions::openai::OpenAICompletionModel> {
                    seedframe::completion::Client::new(
                        #completion_model_init,
                        preamble,
                        1.0,
                        2400,
                        vec![#embedder_instances],
                        #tool_set
                    )
                }
            }
        }
        BuiltInProviderType::DeepseekCompletionModel => {
            let model = config.model.as_ref().unwrap().to_string();
            let completion_model_init = quote! {
                ::seedframe::providers::completions::deepseek::DeepseekCompletionModel::new(std::env::var("SEEDFRAME_DEEPSEEK_API_KEY").unwrap().to_string(), "https://api.deepseek.com/chat/completions".to_string(), #model.to_string())
            };
            quote! {
                #vis async fn build(preamble: String) -> seedframe::completions::Client<::seedframe::providers::completions::deepseek::DeepseekCompletionModel> {
                    seedframe::completion::Client::new(
                        #completion_model_init,
                        preamble,
                        1.0,
                        2400,
                        vec![#embedder_instances],
                        #tool_set
                    )
                }
            }
        }
        BuiltInProviderType::XaiCompletionModel => {
            let model = config.model.as_ref().unwrap().to_string();
            let completion_model_init = quote! {
                ::seedframe::providers::completions::xai::XaiCompletionModel::new(std::env::var("SEEDFRAME_XAI_API_KEY").unwrap().to_string(), "https://api.x.ai/v1/chat/completions".to_string(), #model.to_string())
            };
            quote! {
                #vis async fn build(preamble: String) -> seedframe::completions::Client<::seedframe::providers::completions::xai::XaiCompletionModel> {
                    seedframe::completion::Client::new(
                        #completion_model_init,
                        preamble,
                        1.0,
                        2400,
                        vec![#embedder_instances],
                        #tool_set
                    )
                }
            }
        }
    }
}

fn resolve_tools(tools: &[String]) -> Vec<proc_macro2::Ident> {
    tools
        .iter()
        .map(|tool| {
            let mut parts: Vec<&str> = tool.split("::").collect();
            let last_part = parts.pop().unwrap();
            let name = format!("__SF_TOOL_{}__", last_part);
            parts.push(&name);
            proc_macro2::Ident::new(&parts.join("::"), proc_macro2::Span::call_site())
        })
        .collect()
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

    let provider_type = BuiltInProviderType::from_str(&config.provider)?;
    validate_config(&config, &provider_type)?;

    let tool_execution_mode = if let Some(txm) = &config.clone().execution_mode {
        ExecutionModeType::from_str(&txm)?
    } else {
        ExecutionModeType::FailEarly
    }
    .to_string();
    let tool_execution_mode = syn::Type::from_string(&tool_execution_mode)?;
    let tool_names: Vec<proc_macro2::Ident> =
        resolve_tools(&config.clone().tools.map(|v| v.0).unwrap_or_default());
    let tool_set = quote! { seedframe::tools::ToolSet(vec![#(Box::new(#tool_names::new())),*], #tool_execution_mode) };

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(
        &provider_type,
        &embedder_types,
        &config,
        struct_vis,
        &tool_set,
    );
    let kind: syn::Type = syn::parse_str(&provider_type.to_string()).expect("Failed to parse type");

    Ok(quote! {
        #struct_vis struct #struct_ident{
            inner: seedframe::completion::Client<#kind>,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}
