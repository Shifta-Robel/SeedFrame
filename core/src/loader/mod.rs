use crate::document::Document;
use async_trait::async_trait;
use std::sync::Arc;

use tokio::sync::broadcast::Receiver;

pub mod builtins;

pub(crate) type LoaderInstance = Arc<dyn Loader>;

#[async_trait]
pub trait Loader: Sync {
    async fn subscribe(&self) -> Receiver<Document>;
}
