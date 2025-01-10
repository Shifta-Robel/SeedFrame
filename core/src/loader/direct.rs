use async_trait::async_trait;

use super::LoadedDocument;

#[async_trait]
pub trait DirectLoader {
    async fn retrieve(&self) -> Result<LoadedDocument, DirectLoaderError>;
}

#[derive(Debug, Eq, PartialEq)]
pub enum DirectLoaderError {
    Undefined,
}

#[cfg(test)]
mod tests {
    use crate::loader::{Document, EmbedStrategy, LoadedDocument};

    use super::*;
    use async_trait::async_trait;

    pub struct MyDirectLoader {
        id: uuid::Uuid,
    }

    #[async_trait]
    impl DirectLoader for MyDirectLoader {
        async fn retrieve(&self) -> Result<LoadedDocument, DirectLoaderError> {
            let id = self.id;
            Ok(LoadedDocument {
                document: Document::new_with_id(id, String::from("hello")),
                strategy: EmbedStrategy::IfNotExist,
            })
        }
    }

    #[tokio::test]
    async fn test_simple_direct_loader() {
        let id = uuid::Uuid::new_v4();
        let my_direct_loader = MyDirectLoader { id };
        let res = my_direct_loader.retrieve().await;
        let expected = LoadedDocument {
            document: Document::new_with_id(id, String::from("hello")),
            strategy: EmbedStrategy::IfNotExist,
        };
        assert_eq!(res, Ok(expected));
    }
}
