use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::{Punct, Spacing, TokenStream};
use quote::{format_ident, quote};
use syn::Type;
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct ToolConfig {
    #[darling(default)]
    rename: Option<String>,
}

#[derive(Debug, Error)]
pub(crate) enum ToolMacroError {
    #[error("fn argument '{0}' needs to be documented")]
    UndocumentedArg(String),
    #[error("Description for tool '{0}' not given")]
    DescriptionForFnNotFound(String),
    #[error("Failed to parse tool macro: ")]
    ParseError(#[from] darling::Error),
}

pub(crate) fn tool_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, ToolMacroError> {
    let attr_args: Vec<NestedMeta> = match NestedMeta::parse_meta_list(args) {
        Ok(v) => v,
        Err(e) => {
            return Err(darling::Error::from(e).into());
        }
    };

    let config: ToolConfig = match ToolConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };

    let input: syn::ItemFn = match syn::parse2::<syn::ItemFn>(input) {
        Ok(is) => is,
        Err(e) => {
            return Err(darling::Error::from(e).into());
        }
    };

    validate_fn_type_bounds(&input)?;
    let fn_ident = input.clone().sig.ident;
    let docs: Vec<String> = input
        .attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(meta_name_value) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &meta_name_value.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            return Some(lit_str.value().trim().to_string());
                        }
                    }
                }
            }
            None
        })
        .collect();
    let docs = docs.join("\n");

    let doc_extracted = parse_doc_comment(&docs);
    let (regular_args, state_args) = collect_fn_arg_names(&input)?;

    let mut arg_name_type_desc = Vec::new();
    for arg in regular_args {
        let arg_name = arg.name.clone();
        let arg_type = arg.ty.clone();
        let desc = doc_extracted
            .1
            .iter()
            .find(|v| v.0 == arg_name)
            .ok_or(ToolMacroError::UndocumentedArg(arg_name))?
            .1
            .clone();
        arg_name_type_desc.push((arg.name, arg_type, desc));
    }

    let states = state_args.iter().map(|sa| {
        let ty = &sa.ty;
        quote!{ get_state::<#ty>(states)? }
    });

    let (args, param_struct, params) = get_tool_arg_token_streams(arg_name_type_desc.as_slice())?;

    let fn_call = if input.sig.asyncness.is_some() {
        quote! { #fn_ident(#params, #(#states),*).await }
    } else {
        quote! { #fn_ident(#params, #(#states),*) }
    };

    let tool_name = config.rename.unwrap_or(fn_ident.to_string());
    let description = doc_extracted
        .0
        .ok_or(ToolMacroError::DescriptionForFnNotFound(tool_name.clone()))?;
    let tool_struct_name = format_ident!("__SF_TOOL_{}__", tool_name);
    let m = Punct::new('#', Spacing::Joint);
    let get_state_fn = quote!{
        fn get_state<T: Send + Sync + 'static>(
            states: &dashmap::DashMap<std::any::TypeId, Box<dyn std::any::Any + Send + Sync>>,
        ) -> Result<seedframe::completion::State<T>, seedframe::tools::ToolError> {
            let boxed = states.get(&std::any::TypeId::of::<T>())
                .ok_or(seedframe::tools::ToolError::StateError(seedframe::completion::StateError::NotFound))?;
            
            let arc = boxed.downcast_ref::<std::sync::Arc<T>>()
                .ok_or(seedframe::tools::ToolError::StateError(seedframe::completion::StateError::NotFound))?;
            
            Ok(seedframe::completion::State(arc.clone()))
        }
    };
    Ok(quote! {
        #input
        
        struct #tool_struct_name {
            args: Vec<seedframe::tools::ToolArg>,
        }
        
        impl #tool_struct_name {
            fn new() -> Self {
                Self {
                    args: #args
                }
            }
        }
        
        #m[async_trait::async_trait]
        impl seedframe::tools::Tool for #tool_struct_name {
            fn name(&self) -> &str { &#tool_name }
            fn args(&self) -> &[seedframe::tools::ToolArg] {&self.args}
            fn description(&self) -> &str { &#description }
            async fn call(
                &self,
                args: &str,
                states: &dashmap::DashMap<std::any::TypeId, Box<dyn std::any::Any + Send + Sync>>
            ) -> Result<serde_json::Value, seedframe::tools::ToolError> {
                #get_state_fn
                #param_struct
                let args = args.replace("\\", "");
                let mut args = args.as_str();
                args = &args[1..args.len() - 1];
                let params: Params = serde_json::from_str(args)?;
                Ok(serde_json::to_value(#fn_call)?)
            }
        }
    })
}

fn get_tool_arg_token_streams(
    args: &[(String, Type, String)],
) -> Result<(TokenStream, TokenStream, TokenStream), ToolMacroError> {
    let (a_name, a_type, a_desc) = args.iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut t1, mut t2, mut t3), (a, b, c)| {
            t1.push(a);
            t2.push(b);
            t3.push(c);
            (t1, t2, t3)
        },
    );
    let tool_args = quote! {
        vec![#(seedframe::tools::ToolArg::new::<#a_type>(#a_name, #a_desc),)*]
    };
    let a_name: Vec<proc_macro2::Ident> = a_name
        .iter()
        .map(|n| proc_macro2::Ident::new(n, proc_macro2::Span::call_site()))
        .collect();
    let m = Punct::new('#', Spacing::Joint);
    let params_struct = quote! {
        #m[derive(serde::Deserialize)]
        struct Params {#(#a_name: #a_type,)*}
    };
    let params = quote! {#(params.#a_name),*};

    Ok((tool_args, params_struct, params))
}

fn validate_fn_type_bounds(input: &syn::ItemFn) -> Result<(), darling::Error> {
    for arg in &input.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if parse_state_parameter(pat_type)?.is_some() {
                continue;
            }
            if let syn::Pat::Ident(_) = &*pat_type.pat {
            } else {
                return Err(darling::Error::custom("Regular arguments must be plain identifiers").with_span(&pat_type.pat));
            }
        }
    }
    Ok(())
}

struct RegularArg {
    name: String,
    ty: syn::Type,
}

#[allow(unused)]
#[derive(Clone)]
struct StateArg {
    binding_name: String,
    ty: syn::Type,
}

fn collect_fn_arg_names(input: &syn::ItemFn) -> Result<(Vec<RegularArg>, Vec<StateArg>), darling::Error> {
    let mut regular_args = Vec::new();
    let mut state_args = Vec::new();

    for arg in &input.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let Some((binding_name, inner_ty)) = parse_state_parameter(pat_type)? {
                state_args.push(StateArg { binding_name, ty: inner_ty });
            } else if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    regular_args.push(RegularArg { name: pat_ident.ident.to_string(), ty: *pat_type.ty.clone() });
            } else {
                return Err(darling::Error::custom("Expected identifier pattern for regular argument").with_span(&pat_type.pat));
            }
        } else if let syn::FnArg::Receiver(_) = arg {
            return Err(darling::Error::custom("Methods with 'self' are not supported").with_span(arg));
        }
    }

    Ok((regular_args, state_args))
}

fn parse_state_parameter(pat_type: &syn::PatType) -> Result<Option<(String, syn::Type)>, darling::Error> {
    let ty = match &*pat_type.ty {
        syn::Type::Path(type_path) if type_path.path.segments.len() == 1 => {
            let segment = &type_path.path.segments[0];
            if segment.ident != "State" {
                return Ok(None);
            }
            match &segment.arguments {
                syn::PathArguments::AngleBracketed(args) if args.args.len() == 1 => {
                    if let syn::GenericArgument::Type(inner_ty) = &args.args[0] {
                        inner_ty
                    } else {
                        return Ok(None);
                    }
                }
                _ => return Ok(None),
            }
        }
        _ => return Ok(None),
    };

    let binding_name = match &*pat_type.pat {
        syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
        syn::Pat::TupleStruct(pat_tuple) if pat_tuple.elems.len() == 1 => {
            if let syn::Pat::Ident(pat_ident) = &pat_tuple.elems[0] {
                pat_ident.ident.to_string()
            } else {
                return Ok(None);
            }
        }
        _ => return Ok(None),
    };

    Ok(Some((binding_name, ty.clone())))
}

pub fn parse_doc_comment(doc: &str) -> (Option<String>, Vec<(String, String)>) {
    let lines: Vec<String> = doc
        .lines()
        .map(|line| line.trim_start_matches("///").trim().to_string())
        .collect();

    let mut sections = Vec::new();
    let mut current_section = Vec::new();
    let mut current_heading = None;

    for line in &lines {
        if let Some(stripped) = line.strip_prefix("# ") {
            if !current_section.is_empty() || current_heading.is_some() {
                sections.push((current_heading.take(), current_section));
                current_section = Vec::new();
            }
            current_heading = Some(stripped.trim().to_string());
        } else {
            current_section.push(line.to_string());
        }
    }
    sections.push((current_heading.take(), current_section));

    let description = sections
        .iter()
        .find(|(heading, _)| heading.is_none())
        .map(|(_, lines)| lines.join("\n").trim().to_string());

    let mut arguments = Vec::new();
    if let Some((_, args_lines)) = sections
        .iter()
        .find(|(h, _)| h.as_deref() == Some("Arguments"))
    {
        for line in args_lines {
            if line.starts_with('*') {
                let content = line[1..].trim();
                if let Some((name_part, desc)) = content.split_once(':') {
                    let name = name_part.trim().trim_matches('`').to_string();
                    let description = desc.trim().to_string();
                    arguments.push((name, description));
                }
            }
        }
    }

    (description, arguments)
}
