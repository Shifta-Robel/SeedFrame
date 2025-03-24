use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::{Punct, Spacing, TokenStream};
use quote::{format_ident, quote};
use std::fmt::Display;
use syn::Type;

type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct ToolConfig {
    #[darling(default)]
    rename: Option<String>,
}

#[derive(Debug)]
pub(crate) enum ToolMacroError {
    UndocumentedArg(String),
    DescriptionForFnNotFound(String),
    ParseError(darling::Error),
}

impl From<DarlingError> for ToolMacroError {
    fn from(err: DarlingError) -> Self {
        Self::ParseError(err)
    }
}

impl Display for ToolMacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => {
                write!(f, "Failed to parse tool macro: {e}")
            }
            Self::UndocumentedArg(arg) => {
                write!(f, "fn argument {arg} needs to be documented")
            }
            Self::DescriptionForFnNotFound(tool_name) => {
                write!(f, "Description for tool {tool_name} not given")
            }
        }
    }
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

    let arg_name_type_pairs = collect_fn_arg_names(&input)?;
    let mut arg_name_type_desc: Vec<(String, &Type, String)> = vec![];
    for arg in arg_name_type_pairs {
        let arg_name = arg.0.clone();
        let desc = doc_extracted
            .1
            .iter()
            .find(|v| v.0 == arg_name)
            .ok_or(ToolMacroError::UndocumentedArg(arg_name))?
            .1
            .clone();
        arg_name_type_desc.push((arg.0, arg.1, desc));
    }

    let tool_name = config.rename.unwrap_or(fn_ident.to_string());
    let description = doc_extracted
        .0
        .ok_or(ToolMacroError::DescriptionForFnNotFound(tool_name.clone()))?;

    let m = Punct::new('#', Spacing::Joint);
    let tool_struct_name = format_ident!("__SF_TOOL_{}__", tool_name);
    let (args, param_struct, params) = get_tool_arg_token_streams(&arg_name_type_desc)?;
    let fn_call = if input.sig.asyncness.is_some()  {
        quote! { #fn_ident(#params).await }
    }else {
        quote! { #fn_ident(#params)}
    };
    Ok(quote! {
        #input
        struct #tool_struct_name {
            args: Vec<seedframe::tools::ToolArg>
        }
        impl #tool_struct_name {
            fn new() -> Self {
                Self { args: #args }
            }
        }
        #m[async_trait::async_trait]
        impl seedframe::tools::Tool for #tool_struct_name{
            fn name(&self) -> &str { &#tool_name }
            fn args(&self) -> &[seedframe::tools::ToolArg] {&self.args}
            fn description(&self) -> &str { &#description }
            async fn call(&self, args: &str) -> Result<serde_json::Value, seedframe::tools::ToolError> {
                #param_struct
                let args = args.replace("\\", "");
                let mut args= args.as_str();
                args = &args[1..args.len() - 1];
                let params: Params = serde_json::from_str(args)?;
                Ok(serde_json::to_value(#fn_call)?)
            }
            }
    })
}

fn get_tool_arg_token_streams(
    args: &[(String, &Type, String)],
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
        #m[derive(serde::Deserialize, Debug)]
        struct Params {#(#a_name: #a_type,)*}
    };
    let params = quote! {#(params.#a_name),*};

    Ok((tool_args, params_struct, params))
}

fn validate_fn_type_bounds(input: &syn::ItemFn) -> Result<(), darling::Error> {
    let mut arg_types = Vec::new();

    for arg in &input.sig.inputs {
        match arg {
            syn::FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(_) = &*pat_type.pat {
                } else {
                    return Err(
                        darling::Error::custom("Arguments must be plain identifiers")
                            .with_span(&pat_type.pat),
                    );
                }
                arg_types.push(&pat_type.ty);
            }
            syn::FnArg::Receiver(_) => {
                return Err(
                    darling::Error::custom("Methods with 'self' are not supported").with_span(arg),
                );
            }
        }
    }

    Ok(())
}

fn collect_fn_arg_names(input: &syn::ItemFn) -> Result<Vec<(String, &Type)>, darling::Error> {
    let mut arg_name_type_pairs: Vec<(String, &Type)> = Vec::new();
    for arg in &input.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                arg_name_type_pairs.push((pat_ident.ident.to_string(), &pat_type.ty));
            } else {
                return Err(
                    darling::Error::custom("Expected identifier pattern for argument")
                        .with_span(&pat_type.pat),
                );
            }
        }
    }
    Ok(arg_name_type_pairs)
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
