use async_trait::async_trait;

#[derive(Debug, Clone)]
pub enum Message {
    User(String),
}

type MessageHistory = Vec<Message>;

#[derive(Debug, Clone)]
pub enum CompletionError {
    Undefined,
}

const DEFAULT_TOP_N: usize = 1;

#[async_trait]
pub trait CompletionModel {
    async fn send(
        &self,
        message: Message,
        preamble: &str,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<Message, CompletionError>;
}

pub struct Client<M: CompletionModel> {
    completion_model: M,
    preamble: String,
    history: MessageHistory,

    // common prompt parameters
    temperature: f64,
    max_tokens: usize,

    embedders: Vec<crate::embeddings::Embedder>,
}

impl<M: CompletionModel> Client<M> {
    pub fn new(completion_model: M, preamble: String, temperature: f64, max_tokens: usize) -> Self {
        Self {
            completion_model,
            preamble,
            history: vec![],
            embedders: vec![],
            temperature,
            max_tokens,
        }
    }

    pub async fn prompt(&mut self, prompt: &str) -> Result<Message, CompletionError> {
        let response = self
            .send_prompt(
                prompt,
                &self.preamble,
                &self.history,
                self.temperature,
                self.max_tokens,
            )
            .await;

        if let Ok(ref response) = response {
            self.history.push(response.clone());
        }

        response
    }

    pub async fn one_shot(
        &mut self,
        prompt: &str,
        preamble: Option<String>,
        history: MessageHistory,
    ) -> Result<Message, CompletionError> {
        self.send_prompt(
            prompt,
            &preamble.unwrap_or(self.preamble.clone()),
            &history,
            self.temperature,
            self.max_tokens,
        )
        .await
    }

    async fn send_prompt(
        &self,
        prompt: &str,
        preamble: &str,
        history: &MessageHistory,
        temperature: f64,
        max_tokens: usize,
    ) -> Result<Message, CompletionError> {
        let context = self.get_context(prompt).await;
        let message_with_context =
            Message::User(format!("{prompt}\n\n<context>\n{context}\n</context>\n"));
        let response = self
            .completion_model
            .send(
                message_with_context,
                &preamble,
                history,
                temperature,
                max_tokens,
            )
            .await;

        response
    }

    async fn get_context(&self, prompt: &str) -> String {
        let mut context = String::new();
        for embedder in self.embedders.iter() {
            let query_results = embedder.query(prompt, DEFAULT_TOP_N).await.unwrap();
            for r in query_results {
                context.push_str(&r.raw_data);
            }
        }
        context
    }
}
