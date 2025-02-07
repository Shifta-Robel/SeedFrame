//! Module for loading context from files.
//!
//! Includes loaders for one-time loading (`file_once_loader`) and updating loaders (`file_updating_loader`).

mod utils;

#[allow(dead_code)]
pub mod file_once_loader;

pub mod file_updating_loader;

#[allow(unused)]
pub use file_once_loader::{FileOnceLoaderBuilder, FileOnceLoader};
