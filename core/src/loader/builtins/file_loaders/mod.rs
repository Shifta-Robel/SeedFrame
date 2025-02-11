//! Module for loading context from files.
//!
//! Includes loaders for one-time loading (`file_once_loader`) and updating loaders (`file_updating_loader`).

mod utils;

#[allow(dead_code)]
pub mod file_once_loader;

pub mod file_updating_loader;

#[allow(unused)]
pub use file_once_loader::{FileOnceLoaderBuilder, FileOnceLoader};
use tokio::sync::broadcast::error::SendError;

use crate::document::Document;

#[derive(Debug)]
pub enum FileLoaderError {
    InvalidGlobPattern(glob::PatternError),
    FailedToSendDocument(SendError<Document>),
}

impl From<glob::PatternError> for FileLoaderError{
    fn from(value: glob::PatternError) -> Self {
        Self::InvalidGlobPattern(value)
    }
}
