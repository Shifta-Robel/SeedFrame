use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::fmt::Display;
use thiserror::Error;

#[derive(Debug, FromMeta, Clone)]
struct LoaderConfig {
    #[darling(default)]
    kind: Option<String>,
    #[darling(default)]
    path: Option<String>,
    #[darling(default)]
    external: Option<syn::Type>,
    #[darling(default)]
    config: Option<String>,
}

#[derive(Debug, Error)]
pub(crate) enum LoaderMacroError {
    #[error("Unknown Loader kind: '{0}'. valid options are FileOnceLoader,FileUpdatingLoader")]
    UnknownLoader(String),
    #[error(transparent)]
    ParseError(#[from] darling::Error),
    #[error("Unsupported argument '{0}' for '{1}' loader type")]
    UnsupportedArgument(String, String),
    #[error("Missing required argument '{0}' for '{1}' loader type")]
    MissingArgument(String, String),
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone)]
enum BuiltinLoaderType {
    FileOnceLoader,
    FileUpdatingLoader,
}

impl Display for BuiltinLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::FileOnceLoader =>
                    "seedframe::loader::builtins::file_loaders::file_once_loader::FileOnceLoader",
                Self::FileUpdatingLoader =>
                    "seedframe::loader::builtins::file_loaders::file_updating_loader::FileUpdatingLoader",
            }
        )
    }
}

impl BuiltinLoaderType {
    fn from_str(kind: &str) -> Result<Self, LoaderMacroError> {
        match kind {
            "FileOnceLoader" => Ok(Self::FileOnceLoader),
            "FileUpdatingLoader" => Ok(Self::FileUpdatingLoader),
            unknown => Err(LoaderMacroError::UnknownLoader(unknown.to_string())),
        }
    }
    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::FileOnceLoader | Self::FileUpdatingLoader => &["path"],
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

    validate_config(&config)?;
    let (struct_ident, struct_vis) = (&input.ident, &input.vis);

    let static_loader_instance_ident =
        format_ident!("__{}_INSTANCE", struct_ident.to_string().to_uppercase(),);
    let static_loader_instance = quote! {
        static #static_loader_instance_ident: ::std::sync::LazyLock<::std::sync::Arc<#struct_ident>>
          = ::std::sync::LazyLock::new(||{
              ::std::sync::Arc::new(#struct_ident::build())
        });
    };

    let loader_type = get_type(&config)?;
    let builder_impl = generate_builder(&config, &loader_type, struct_vis);

    let kind = match loader_type {
        LoaderType::BuiltIn(t) => t,
        LoaderType::External(t) => t,
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

enum LoaderType {
    BuiltIn(syn::Type),
    External(syn::Type),
}

fn validate_config(config: &LoaderConfig) -> Result<(), LoaderMacroError> {
    if config.kind.is_some() && config.external.is_some() {
        Err(LoaderMacroError::ParseError(
                darling::Error::custom(
                    "Only one of the attributes `kind` or `external` is supported!")))?
    }
    if config.kind.is_none() && config.external.is_none() {
        Err(LoaderMacroError::ParseError(
                darling::Error::custom(
                    "Macro expects one of `kind` or `external` attributes to be specified!")))?
    }

    if let Some(kind) = &config.kind {
        let loader_type = BuiltinLoaderType::from_str(kind)?;
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
            }
            else {
                Ok(())
            }
        };
        check_arg("path", &config.path)?;
    }else if config.path.is_some() {
        Err(LoaderMacroError::UnsupportedArgument( "path".to_string(), "external".to_string(),))?
    };
    Ok(())
}

fn get_type(config: &LoaderConfig) -> Result<LoaderType, LoaderMacroError> {
    if let Some(kind) = &config.kind {
        Ok(LoaderType::BuiltIn(
            syn::Type::from_string(&BuiltinLoaderType::from_str(kind)?.to_string())?
        ))
    }else {
        Ok(LoaderType::External(
                config.external.clone().unwrap()
        ))
    }
}

fn generate_builder(
    config: &LoaderConfig,
    loader_type: &LoaderType,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
    match loader_type {
        LoaderType::BuiltIn(t) => {
            let path = config.path.as_ref().unwrap().to_string();
            quote! {
                #vis fn build() -> Self {
                    Self { inner: (#t::new(vec![#path.to_string()]).unwrap().build().unwrap()) }
                }
            }
        },
        LoaderType::External(t) => {
            if let Some(json_str) = &config.config {
                quote! {
                    #vis fn build() -> Self {
                        Self { inner: (#t::new(Some(#json_str)).unwrap()) }
                    }
                }
            }else {
                quote! {
                    #vis fn build() -> Self {
                        Self { inner: (#t::new(None).unwrap()) }
                    }
                }
            }
        }
    }
}
