pub mod direct;
pub mod publishing;

use crate::{document::Document, embeddings::EmbeddingUpdateStrategy};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LoadedDocument {
    pub document: Document,
    pub strategy: EmbeddingUpdateStrategy,
}

impl LoadedDocument {
    pub fn new(document: Document, strategy: EmbeddingUpdateStrategy) -> Self {
        Self { document, strategy }
    }
}
