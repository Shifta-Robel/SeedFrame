use proc_macro2::TokenStream;
use darling::FromMeta;
use quote::{format_ident, quote};

// #[tool(
//     rename: "adder",
//     args: [
//         ("a", "The first number"),
//         ("b", "The second number"),]
//     )]
// /// A function to add two numbers
// ///
// /// # Arguments
// /// * `a`: The first number
// /// * `b`: The second number
// pub fn add(a: u32, b: u32) -> u32 {
//     a + b
// }

// type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct ToolConfig {
    #[darling(default)]
    _rename: Option<String>
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum ToolMacroError {
    UnknownTool(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
}

pub(crate) fn tool_impl(
    _args: TokenStream,
    _input: TokenStream,
) -> Result<TokenStream, ToolMacroError> {
    let given_tool_name = "";
    let tool_name = format_ident!("SF_TOOL_{}", given_tool_name);
    Ok(
        quote! {
            struct #tool_name;
            impl seedframe::tool::Tool for #tool_name {
                pub fn description() -> seedframe::tool::ToolDescription {
                    ToolDescription {
                        name: #given_tool_name,
                        description: *given_tool_desc,
                        args: 
                    }
                }
            }
        }
    )
}
