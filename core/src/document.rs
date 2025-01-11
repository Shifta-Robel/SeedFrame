#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Document {
    pub id: usize,
    pub data: String,
}

impl Document {
    pub fn new(data: String) -> Self {
        Self {
            // id: uuid::Uuid::new_v4(),
            id: 2,
            data,
        }
    }

    pub fn new_with_id(id: usize, data: String) -> Self {
        Self { id, data }
    }
}
