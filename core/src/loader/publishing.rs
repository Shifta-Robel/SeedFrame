use async_trait::async_trait;
use tokio::sync::broadcast;

use super::LoadedDocument;

#[async_trait]
pub trait PublishingLoader {
    async fn subscribe(&'_ self) -> broadcast::Receiver<LoadedDocument>;
}

#[cfg(test)]
mod tests {
    use crate::{document::Document, embeddings::EmbeddingUpdateStrategy};

    use super::*;
    use async_trait::async_trait;

    pub struct MyPublishingLoader {
        tx: broadcast::Sender<LoadedDocument>,
    }

    impl MyPublishingLoader {
        pub async fn init() -> Self {
            let (tx, mut _rx) = broadcast::channel(32);
            let txc = tx.clone();
            tokio::spawn(async move {
                loop {
                    let id = 0;
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    _ = txc
                        .send(LoadedDocument {
                            document: Document::new_with_id(id, String::from("hello")),
                            strategy: EmbeddingUpdateStrategy::AppendAsNew,
                        })
                        .unwrap();
                }
            });
            Self { tx }
        }
    }

    #[async_trait]
    impl PublishingLoader for MyPublishingLoader {
        async fn subscribe(&'_ self) -> broadcast::Receiver<LoadedDocument> {
            self.tx.subscribe()
        }
    }

    #[tokio::test]
    async fn test_simple_publishing_loader() {
        let mut msgs = vec![];

        _ = tokio::time::timeout(tokio::time::Duration::from_millis(350), async {
            let my_publishing_loader = MyPublishingLoader::init().await;
            let mut my_recv = my_publishing_loader.subscribe().await;

            while let Ok(s) = my_recv.recv().await {
                msgs.push(s);
            }
        })
        .await;

        let expected = LoadedDocument {
            document: Document::new_with_id(0, String::from("hello")),
            strategy: EmbeddingUpdateStrategy::AppendAsNew,
        };
        assert_eq!(vec![expected.clone(), expected.clone(), expected], msgs);
    }
}
