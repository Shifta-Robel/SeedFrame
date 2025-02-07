#[derive(Clone, Debug, PartialEq)]
/// Embedding of a document
pub struct Embedding {
    /// A unique identifier for the embedding.
    pub id: String,
    /// The numerical embedding vector of the document's content.
    pub embedded_data: Vec<f64>,
    /// The raw text data from which the embedding was generated.
    pub raw_data: String,
}
