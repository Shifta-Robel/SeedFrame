use crate::document::Document;
use async_trait::async_trait;
use tokio::sync::broadcast::Receiver;

pub mod builtins;

#[async_trait]
pub trait Loader: Sync {
    async fn subscribe(&self) -> Receiver<Document>;
}
