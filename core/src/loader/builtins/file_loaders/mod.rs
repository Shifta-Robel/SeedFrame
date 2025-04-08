//! Module for loading context from files.
//!
//! Includes loaders for one-time loading (`file_once_loader`) and updating loaders (`file_updating_loader`).

mod utils;

#[allow(dead_code)]
pub mod file_once_loader;

pub mod file_updating_loader;

#[allow(unused)]
pub use file_once_loader::{FileOnceLoader, FileOnceLoaderBuilder};
use thiserror::Error;
use tokio::sync::broadcast::error::SendError;

use crate::document::Document;

#[derive(Debug, Error)]
pub enum FileLoaderError {
    #[error("Invalid glob-pattern")]
    InvalidGlobPattern(#[from] glob::PatternError),
    #[error("No matching documents found")]
    NoMatchingDocuments,
    #[error("Failed to send loaded document")]
    FailedToSendDocument(#[from] SendError<Document>),
}
