use async_trait::async_trait;
use tokio::sync::broadcast;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{document::Document, loader::Loader};
use super::utils::{parse_file, resolve_input_to_files};

pub struct FileOnceLoaderBuilder {
    paths: Vec<String>,
}

impl FileOnceLoaderBuilder {
    pub fn new(paths: Vec<String>) -> Self {
        Self { paths }
    }

    pub fn build(self) -> FileOnceLoader {
        let files = resolve_input_to_files(self.paths.iter().map(|s| s.as_str()).collect()).unwrap();
        let (tx, _rx) = broadcast::channel(files.len());

        let mut documents: Vec<Document> = vec![];
        for file in files {
            let data = parse_file(&file).unwrap();
            let document = Document {
                id: file.to_string_lossy().to_string(),
                data,
            };
            documents.push(document);
        }

        FileOnceLoader {
            tx,
            documents,
            sent: AtomicBool::new(false),
        }
    }
}

pub struct FileOnceLoader {
    tx: broadcast::Sender<Document>,
    documents: Vec<Document>,
    sent: AtomicBool,
}

#[async_trait]
impl Loader for FileOnceLoader {
    async fn subscribe(&self) -> broadcast::Receiver<Document> {
        let receiver = self.tx.subscribe();
        if !self.sent.load(Ordering::Acquire) {
            if self.sent.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                for doc in &self.documents {
                    self.tx.send(doc.clone()).unwrap();
                }
            }
        }
        receiver
    }
}
