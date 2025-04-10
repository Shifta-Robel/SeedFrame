/// Represents contents of a document for use in embedding,
/// and similarity search.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Document {
    /// identifies a document in the store
    pub id: String,
    /// raw data of the document
    pub data: String,
}

impl Document {
    #[must_use] pub fn new(id: String, data: String) -> Self {
        Self { id, data }
    }
}
