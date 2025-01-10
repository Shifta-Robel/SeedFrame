pub mod direct;
pub mod publishing;

#[derive(Debug, PartialEq, Eq)]
pub struct Document {
    pub content: Vec<String>,
    pub embed_strategy: EmbedStrategy
}

#[derive(Debug, PartialEq, Eq)]
pub enum EmbedStrategy {
    Refresh,
    DontRefresh
}
