use async_trait::async_trait;

use super::Document;

#[async_trait]
pub trait DirectLoader {
    async fn retrieve(self) -> Document;
}

#[cfg(test)]
mod tests {
    use crate::loader::{Document, EmbedStrategy};

    use super::*;
    use async_trait::async_trait;

    pub struct MyDirectLoader;

    #[async_trait]
    impl DirectLoader for MyDirectLoader {
        async fn retrieve(self) -> Document {
            Document {
                content: vec![String::from("hello world")],
                embed_strategy: EmbedStrategy::DontRefresh
            }
        }
    }

    #[tokio::test]
    async fn test_simple_direct_loader() {
        let my_direct_loader = MyDirectLoader;
        let res = my_direct_loader.retrieve().await;
        let doc = Document {
                content: vec![String::from("hello world")],
                embed_strategy: EmbedStrategy::DontRefresh
        };
        assert_eq!(res, doc);
    }
}
