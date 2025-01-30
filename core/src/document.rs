#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Document {
    pub id: String,
    pub data: String,
}

impl Document {
    pub fn new(id: String, data: String) -> Self {
        Self { id, data }
    }
}
