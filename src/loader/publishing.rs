use async_trait::async_trait;
use tokio::sync::broadcast;

#[async_trait]
pub trait PublishingLoader {
    async fn subscribe(&'_ self) -> broadcast::Receiver<String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    pub struct MyPublishingLoader {
        tx: broadcast::Sender<String>,
    }

    impl MyPublishingLoader {
        pub async fn init() -> Self {
            let (tx, mut _rx) = broadcast::channel(32);
            let txc = tx.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    txc.send(String::from("Hi")).unwrap();
                }
            });
            Self { tx }
        }
    }

    #[async_trait]
    impl PublishingLoader for MyPublishingLoader {
        async fn subscribe(&'_ self) -> broadcast::Receiver<String> {
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

        assert_eq!(
            vec![String::from("Hi"), String::from("Hi"), String::from("Hi")],
            msgs
        );
    }
}
