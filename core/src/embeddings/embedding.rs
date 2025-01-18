use uuid::Uuid;

pub struct Embedding {
    pub id: Uuid,
    pub embedded_data: Vec<f64>,
    pub raw_data: String,
}
