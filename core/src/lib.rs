pub mod embeddings;
pub mod loader;
pub mod vector_store;
pub mod providers;

pub mod document {
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct Document {
        pub id: uuid::Uuid,
        pub data: String,
    }

    impl Document {
        pub fn new(data: String) -> Self {
            Self {
                id: uuid::Uuid::new_v4(),
                data,
            }
        }

        pub fn new_with_id(id: uuid::Uuid, data: String) -> Self {
            Self { id, data }
        }
    }
}
