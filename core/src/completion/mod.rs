use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct Message;
type MessageHistory = Vec<Message>;

#[derive(Debug, Clone)]
pub struct ResponseMessage;

#[derive(Debug, Clone)]
pub enum CompletionError {
    Undefined,
}

#[async_trait]
pub trait CompletionModel {
    async fn send(
        &self,
        message: Message,
        history: MessageHistory,
    ) -> Result<ResponseMessage, CompletionError>;
}

pub struct Client<M: CompletionModel> {
    completion_model: M,
    preamble: String,
    history: MessageHistory,
    embedders: Vec<crate::embeddings::Embedder>, 
}

impl<M: CompletionModel> Client<M> {
    pub fn new(completion_model: M, preamble: String) -> Self {
        Self {
            completion_model,
            preamble,
            history: vec![],
            embedders: vec![],
        }
    }

}
