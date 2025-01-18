use crate::document::Document;
use async_trait::async_trait;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Defines wether the loader will be checking for changes or not
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum LoadingStrategy {
    /// Loaded resource is assumed to be static, loader will load the resource once and not check for updates
    Static,
    /// Loaded resource is assumed to change over time
    Dynamic,
}

#[async_trait]
pub trait Loader {
    fn strategy(&self) -> LoadingStrategy;
    fn id(&self) -> Uuid;

    async fn subscribe(&'_ self) -> broadcast::Receiver<Document>;
}
