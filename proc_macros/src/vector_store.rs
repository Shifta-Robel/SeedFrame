use darling::{ast::NestedMeta, FromMeta};
use proc_macro2::TokenStream;
use quote::quote;
use std::fmt::Display;

type DarlingError = darling::Error;

#[derive(Debug, FromMeta, Clone)]
struct VectorStoreConfig {
    kind: String,
    #[darling(default)]
    host: Option<String>,
    #[darling(default)]
    namespace: Option<String>,
    #[darling(default)]
    env_var: Option<String>,
    #[darling(default)]
    source_tag: Option<String>,
}

#[allow(unused)]
#[derive(Debug)]
pub(crate) enum VectorStoreMacroError {
    UnknownVectorStore(String),
    ParseError(darling::Error),
    UnsupportedArgument(String, String),
    MissingArgument(String, String),
}

impl From<DarlingError> for VectorStoreMacroError {
    fn from(err: DarlingError) -> Self {
        Self::ParseError(err)
    }
}

impl Display for VectorStoreMacroError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => {
                write!(f, "Failed to parse vector_store macro: {e}")
            }
            Self::UnknownVectorStore(l) => {
                write!(
                    f,
                    "Unknown VectorStore kind: '{l}'. valid options are InMemoryVectorStore"
                )
            }
            Self::UnsupportedArgument(arg, vector_store) => {
                write!(
                    f,
                    "Unsupported argument '{arg}' for '{vector_store}' vector store type"
                )
            }
            Self::MissingArgument(arg, vector_store) => {
                write!(
                    f,
                    "Missing required argument '{arg}' for '{vector_store}' vector_store type"
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
enum VectorStoreType {
    InMemoryVectorStore,
    Pinecone,
}

impl Display for VectorStoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::InMemoryVectorStore =>
                    "seedframe::vector_store::in_memory_vec_store::InMemoryVectorStore",
                Self::Pinecone => "seedframe::vector_store::pinecone::PineconeVectorStore",
            }
        )
    }
}

impl VectorStoreType {
    fn from_str(kind: &str) -> Result<Self, VectorStoreMacroError> {
        match kind {
            "InMemoryVectorStore" => Ok(Self::InMemoryVectorStore),
            "pinecone" => Ok(Self::Pinecone),
            unknown => Err(VectorStoreMacroError::UnknownVectorStore(
                unknown.to_string(),
            )),
        }
    }

    fn required_args(&self) -> &'static [&'static str] {
        match self {
            Self::InMemoryVectorStore => &[],
            Self::Pinecone => &["host"],
        }
    }

    fn supported_args(&self) -> &'static [&'static str] {
        match self {
            Self::InMemoryVectorStore => &[],
            Self::Pinecone => &["host", "namespace", "env_var", "source_tag"],
        }
    }
}

fn validate_config(
    config: &VectorStoreConfig,
    vector_store_type: &VectorStoreType,
) -> Result<(), VectorStoreMacroError> {
    let required = vector_store_type.required_args();
    let supported = vector_store_type.supported_args();
    let check_arg = |name: &str, value: &Option<String>| {
        if value.is_none() && required.contains(&name) {
            Err(VectorStoreMacroError::MissingArgument(
                name.to_string(),
                vector_store_type.to_string(),
            ))
        } else if value.is_some() && !supported.contains(&name) {
            Err(VectorStoreMacroError::UnsupportedArgument(
                name.to_string(),
                vector_store_type.to_string(),
            ))
        } else {
            Ok(())
        }
    };

    _ = check_arg("host", &config.host)?;
    _ = check_arg("namespace", &config.namespace)?;
    _ = check_arg("env_var", &config.env_var)?;
    _ = check_arg("source_tag", &config.source_tag)?;
    Ok(())
}

fn generate_builder(
    vector_store_type: &VectorStoreType,
    config: &VectorStoreConfig,
    vis: &syn::Visibility,
) -> proc_macro2::TokenStream {
    match vector_store_type {
        VectorStoreType::InMemoryVectorStore => {
            quote! {
                #vis async fn build() -> Result<Self, seedframe::vector_store::VectorStoreError> {
                    Ok(Self {
                        inner: (seedframe::vector_store::in_memory_vec_store::InMemoryVectorStore::new())
                    })
                }
            }
        }
        VectorStoreType::Pinecone => {
            let host: &str = config.host.as_ref().unwrap();
            let env: Option<String> = config.env_var.clone();
            let source_tag: Option<String> = config.source_tag.clone();
            let namespace: Option<String> = config.namespace.clone();
            //
            let host_expr =
                syn::parse_str::<syn::Expr>(&format!("\"{host}\".to_string()")).unwrap();
            let env_expr = option_expr(env);
            let source_tag_expr = option_expr(source_tag);
            let namespace_expr = option_expr(namespace);
            quote! {
                #vis async fn build() -> Result<Self, seedframe::vector_store::VectorStoreError> {
                    Ok(Self {
                        inner: PineconeVectorStore::new(#env_expr, #host_expr, #source_tag_expr, #namespace_expr).await?,
                    })
                }
            }
        }
    }
}

fn option_expr(opt: Option<String>) -> syn::Expr {
    let expr = if let Some(v) = opt {
        &format!("Some(\"{}\".to_string())", v)
    } else {
        "None"
    };
    syn::parse_str(expr).unwrap()
}

pub(crate) fn vector_store_impl(
    args: TokenStream,
    input: TokenStream,
) -> Result<TokenStream, VectorStoreMacroError> {
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

    let config: VectorStoreConfig = match VectorStoreConfig::from_list(&attr_args) {
        Ok(v) => v,
        Err(e) => {
            return Err(e.into());
        }
    };

    let vector_store_type = VectorStoreType::from_str(&config.kind)?;
    validate_config(&config, &vector_store_type)?;

    let (struct_ident, struct_vis) = (&input.ident, &input.vis);
    let builder_impl = generate_builder(&vector_store_type, &config, &struct_vis);
    let kind: syn::Type =
        syn::parse_str(&vector_store_type.to_string()).expect("Failed to parse type");

    Ok(quote! {
        #struct_vis struct #struct_ident{
            pub inner: #kind,
        }

        impl #struct_ident {
            #builder_impl
        }
    })
}
