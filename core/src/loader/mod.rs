use crate::document::Document;
use async_trait::async_trait;
use std::sync::Arc;

use tokio::sync::broadcast::Receiver;

/// Module for built-in loader implementations.
///
/// Provides pre-defined loader implementations that can be used
/// for common resource loading.
pub mod builtins;

pub(crate) type LoaderInstance = Arc<dyn Loader>;

/// A trait for resource loaders.
///
/// Defines the interface for loaders.
/// Implementations of this trait are responsible for loading resources and publishing
/// them to a broadcast channel.
#[async_trait]
pub trait Loader: Sync {
    async fn subscribe(&self) -> Receiver<Document>;
}
