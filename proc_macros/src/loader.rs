use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::fmt::Display;
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct LoaderConfig {
    kind: String,
    #[darling(default)]
    url: Option<String>,
    #[darling(default)]
    path: Option<String>,
    #[darling(default)]
    interval: Option<u64>,
}

#[derive(Debug, Error)]
pub(crate) enum LoaderMacroError {
    #[error("Unknown Loader kind: '{0}'. valid options are FileOnceLoader,FileUpdatingLoader,HttpOnceLoader")]
    UnknownLoader(String),
    #[error("Failed to parse loader macro: ")]
    ParseError(#[from] darling::Error),
    #[error("Unsupported argument '{0}' for '{1}' loader type")]
    UnsupportedArgument(String, String),
    #[error("Missing required argument '{0}' for '{1}' loader type")]
    MissingArgument(String, String),
}

#[derive(Debug, Clone)]
enum BuiltinLoaderType {
    FileOnceLoader,
    FileUpdatingLoader,
    HttpOnceLoader,
}

impl Display for BuiltinLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::FileOnceLoader => "seedframe::loader::builtins::file_loaders::file_once_loader::FileOnceLoader",
                Self::FileUpdatingLoader => "seedframe::loader::builtins::file_loaders::file_updating_loader::FileUpdatingLoader",
                Self::HttpOnceLoader => "HttpOnceLoader",
            }
        )
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

    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
            Self::HttpOnceLoader => &["url"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
            Self::HttpOnceLoader => &["url"],
        }
    }
}

fn validate_config(
    config: &LoaderConfig,
    loader_type: &BuiltinLoaderType,
) -> Result<(), LoaderMacroError> {
    let required = loader_type.required_args();
    let supported = loader_type.supported_args();
    let check_arg = |name: &str, value: &Option<String>| {
        if value.is_none() && required.contains(&name) {
            Err(LoaderMacroError::MissingArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        } else if value.is_some() && !supported.contains(&name) {
            Err(LoaderMacroError::UnsupportedArgument(
                name.to_string(),
                loader_type.to_string(),
            ))
        } else {
            Ok(())
        }
    };
    check_arg("path", &config.path)?;
    check_arg("url", &config.url)?;
    check_arg("interval", &config.interval.map(|v| v.to_string()))?;

    Ok(())
}

fn generate_builder(
    loader_type: &BuiltinLoaderType,
    config: &LoaderConfig,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
    match loader_type {
        BuiltinLoaderType::FileOnceLoader => {
            let path = config.path.as_ref().unwrap().to_string();
            quote! {
                #vis fn build() -> Self {
                    Self { inner: (seedframe::loader::builtins::file_loaders::file_once_loader::FileOnceLoaderBuilder::new(vec![#path.to_string()]).unwrap().build().unwrap()) }
                }
            }
        }
        BuiltinLoaderType::FileUpdatingLoader => {
            let path = config.path.as_ref().unwrap().to_string();
            quote! {
                #vis fn build() -> Self {
                    Self { inner: (seedframe::loader::builtins::file_loaders::file_updating_loader::FileUpdatingLoaderBuilder::new(vec![#path.to_string()]).unwrap().build().unwrap()) }
                }
            }
        }
        BuiltinLoaderType::HttpOnceLoader => {
            quote! {}
        }
    }
}

pub(crate) fn loader_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, LoaderMacroError> {
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

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(&loader_type, &config, struct_vis);
    let kind: syn::Type = syn::parse_str(&loader_type.to_string()).expect("Failed to parse type");

    let static_loader_instance_ident =
        format_ident!("__{}_INSTANCE", struct_ident.to_string().to_uppercase(),);
    let static_loader_instance = quote! {
        static #static_loader_instance_ident: ::std::sync::LazyLock<::std::sync::Arc<#struct_ident>>
          = ::std::sync::LazyLock::new(||{
              ::std::sync::Arc::new(#struct_ident::build())
        });
    };

    Ok(quote! {
        #struct_vis struct #struct_ident{
            pub inner: #kind,
        }

        impl #struct_ident {
            #builder_impl
        }

        #[async_trait::async_trait]
        impl ::seedframe::loader::Loader for #struct_ident {
            async fn subscribe(&self) -> ::tokio::sync::broadcast::Receiver<::seedframe::document::Document> {
                self.inner.subscribe().await
            }
        }

        #static_loader_instance
    })
}
