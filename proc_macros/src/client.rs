use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use std::{fmt::Display, str::FromStr};
use syn::{parse::Parser, ItemStruct, Meta};
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct ClientConfig {
    #[darling(default)]
    provider: Option<String>,
    #[darling(default)]
    model: Option<String>,
    #[darling(default)]
    tools: Option<ToolNames>,
    #[darling(default)]
    execution_mode: Option<String>,
    #[darling(default)]
    external: Option<String>,
    #[darling(default)]
    api_key: Option<String>,
    #[darling(default)]
    url: Option<String>
}

#[derive(Debug, Error)]
pub(crate) enum ClientMacroError {
    #[error("Unknown completion model provider: '{0}'. valid options are openai, deepseek, xai")]
    UnknownCompletionModel(String),
    #[error("Failed to parse client macro: ")]
    ParseError(#[from] darling::Error),
    #[error("Unsupported argument '{0}' for '{1}' client type")]
    UnsupportedArgument(String, String),
    #[error("Missing required argument '{0}' for '{1}' client type")]
    MissingArgument(String, String),
    #[error("Unrecognized attribute {0}")]
    UnrecognizedAttribute(String),
    #[error("Unknown execution mode : '{0}'. valid options are fail_early or best_effort")]
    UnknownExecutionMode(String),
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
                    .map_err(darling::Error::from)?;
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

impl FromStr for BuiltInProviderType {
    type Err = ClientMacroError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "openai" => Ok(Self::OpenAICompletionModel),
            "deepseek" => Ok(Self::DeepseekCompletionModel),
            "xai" => Ok(Self::XaiCompletionModel),
            unknown => Err(ClientMacroError::UnknownCompletionModel(
                unknown.to_string(),
            )),
        }
    }
}

fn validate_config(config: &ClientConfig) -> Result<(), ClientMacroError> {
    Ok(())
}

enum ProviderType {
    BuiltIn(syn::Type),
    External(syn::Type)
}

fn parse_completion_provider(config: &ClientConfig) -> Result<ProviderType, ClientMacroError> {
    validate_config(config)?;
    let type_str = if let Some(p) = &config.provider {
        BuiltInProviderType::from_str(&p)?.to_string()
    }else {
        config.external.clone().unwrap()
    };
    let p_type = syn::parse_str(&type_str).map_err(|e| ClientMacroError::ParseError(darling::Error::from(e)))?;

    if config.provider.is_some() {
        Ok(ProviderType::BuiltIn(p_type))
    }else {
        Ok(ProviderType::External(p_type))
    }
}

fn parse_tools(tools: &[String]) -> Vec<proc_macro2::Ident> {
    tools
        .iter()
        .map(|tool| {
            let mut parts: Vec<&str> = tool.split("::").collect();
            let last_part = parts.pop().unwrap();
            let name = format!("__SF_TOOL_{last_part}__");
            parts.push(&name);
            proc_macro2::Ident::new(&parts.join("::"), proc_macro2::Span::call_site())
        })
        .collect()
}

fn parse_embedders(
    input: &ItemStruct,
) ->Result<TokenStream, ClientMacroError> {
    let embedder_types = {
        let mut embedders = Vec::new();
        for f in input.clone().fields {
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

    let mut embedder_instances = quote! {};
    for embedder_type in embedder_types {
        embedder_instances.extend(quote! {
            #embedder_type::build().await.inner,
        });
    }
    Ok(embedder_instances)
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

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let embedder_instances = parse_embedders(&input)?;

    let tool_execution_mode = if let Some(txm) = &config.clone().execution_mode {
        ExecutionModeType::from_str(txm)?
    } else {
        ExecutionModeType::FailEarly
    }
    .to_string();
    let tool_execution_mode = syn::Type::from_string(&tool_execution_mode)?;
    let tool_names: Vec<proc_macro2::Ident> =
        parse_tools(&config.clone().tools.map(|v| v.0).unwrap_or_default());
    let tool_set = quote! { seedframe::tools::ToolSet(vec![#(Box::new(#tool_names::new())),*], #tool_execution_mode) };

    let model = config.model.as_ref().unwrap().to_string();
    let kind: syn::Type;
    let model_init = match parse_completion_provider(&config)? {
        ProviderType::BuiltIn(t) => {
            kind = t.clone();
            quote! {let model = #t::new(None, None, String::from(#model));}
        },
        ProviderType::External(t) => {
            kind = t.clone();
            quote! {let model = #t::new();}
        }
    };

    Ok(
        quote! {
            #struct_vis struct #struct_ident;
            
            impl #struct_ident{
                #struct_vis async fn build(preamble: impl AsRef<str>) -> seedframe::completion::Client<#kind> {
                    use seedframe::completion::CompletionModel;
                    #model_init
                    model.build_client(
                        preamble,
                        vec![#embedder_instances],
                        #tool_set
                    )
                }
            }

        }
    )
}
