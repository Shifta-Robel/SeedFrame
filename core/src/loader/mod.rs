pub mod direct;
pub mod publishing;

use crate::document::Document;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EmbedStrategy {
    IfNotExist,
    Refresh,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LoadedDocument {
    pub document: Document,
    pub strategy: EmbedStrategy,
}

impl LoadedDocument {
    pub fn new(document: Document, strategy: EmbedStrategy) -> Self {
        Self { document, strategy }
    }
}
