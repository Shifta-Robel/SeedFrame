use proc_macro2::TokenStream;
use darling::{ast::NestedMeta, FromMeta};
use quote::quote;
use std::fmt::Display;

type DarlingError = darling::Error;

#[derive(Debug,FromMeta, Clone)]
struct LoaderConfig{
    kind: String,
    #[darling(default)]
    url: Option<String>,
    #[darling(default)]
    path: Option<String>,
    #[darling(default)]
    interval: Option<u64>,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum LoaderMacroError {
    UnknownLoader(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
}

impl From<DarlingError> for LoaderMacroError {
    fn from(err: DarlingError) -> Self {
        Self::ParseError(err)
    }
}

impl Display for LoaderMacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => {
                write!(f, "Failed to parse loader macro: {e}")
            },
            Self::UnknownLoader(l) => {
                write!(f, "Unknown Loader kind: '{l}'. valid optionss are FileOnceLoader, FileUpdatingLoader, HttpOnceLoader")
            },
            Self::UnsupportedArgument(arg, loader) => {
                write!(f, "Unsupported argument '{arg}' for '{loader}' loader type")
            },
            Self::MissingArgument(arg, loader) => {
                write!(f, "Missing required argument '{arg}' for '{loader}' loader type")
            },
        }
    }
}

#[derive(Debug,Clone)]
enum BuiltinLoaderType {
    FileOnceLoader,
    FileUpdatingLoader,
    HttpOnceLoader
}

impl Display for BuiltinLoaderType{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"{}", match self {
            Self::FileOnceLoader => "FileOnceLoader",
            Self::FileUpdatingLoader => "FileUpdatingLoader",
            Self::HttpOnceLoader => "HttpOnceLoader"
        })
    }
}

impl BuiltinLoaderType {
    fn from_str(kind: &str) -> Result<Self, LoaderMacroError> {
        match kind {
            "FileOnceLoader" => Ok(Self::FileOnceLoader),
            "FileUpdatingLoader" => Ok(Self::FileUpdatingLoader),
            "HttpOnceLoader" => Ok(Self::HttpOnceLoader),
            unknown => Err(LoaderMacroError::UnknownLoader(unknown.to_string())),
        }
    }

    fn required_args(&self) ->  &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
            Self::HttpOnceLoader =>  &["url"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
            Self::HttpOnceLoader => &["url"],
        }
    }
}

fn validate_config(config: &LoaderConfig, loader_type: &BuiltinLoaderType) -> Result<(), LoaderMacroError> {
    let required = loader_type.required_args();
    let supported = loader_type.supported_args();
    let check_arg = |name: &str, value: &Option<String>| {
        if value.is_none() && required.contains(&name) {
            Err(LoaderMacroError::MissingArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        }else if value.is_some() && !supported.contains(&name) {
            Err(LoaderMacroError::UnsupportedArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        }else {
            Ok(())
        }
    };
    _ = check_arg("path", &config.path);
    _ = check_arg("url", &config.url);
    _ = check_arg("interval", &config.interval.map(|v| v.to_string()));

    Ok(())
}

fn generate_builder(loader_type: &BuiltinLoaderType, config: &LoaderConfig) -> proc_macro2::TokenStream {
    match loader_type {
        BuiltinLoaderType::FileOnceLoader => {
            let path = config.path.as_ref().unwrap();
            quote! {
                pub(crate) fn build() -> Self {
                    Self { inner: FileOnceLoaderBuilder::new(#path) }
                }
            }
        }
        BuiltinLoaderType::FileUpdatingLoader => {
            quote! { }
        }
        BuiltinLoaderType::HttpOnceLoader => {
            quote! { }
        }
    }
}

pub(crate) fn loader_impl(args: TokenStream, input: TokenStream) -> Result<TokenStream, LoaderMacroError> {
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
    let config: LoaderConfig = match LoaderConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };

    let loader_type = BuiltinLoaderType::from_str(&config.kind)?;
    validate_config(&config, &loader_type)?;

    let input_struct: syn::ItemStruct = input.clone();
    let struct_ident = &input_struct.ident;
    let builder_impl = generate_builder(&loader_type, &config);

    Ok(quote! {
        #input_struct

        impl #struct_ident {
            #builder_impl
        }
    })
}
