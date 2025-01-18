use uuid::Uuid;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Document {
    pub id: Uuid,
    pub data: String,
}

impl Document {
    pub fn new(id: Uuid, data: String) -> Self {
        Self { id, data }
    }
}
