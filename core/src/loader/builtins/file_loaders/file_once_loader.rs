use async_trait::async_trait;
use tokio::sync::broadcast;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{document::Document, loader::Loader};
use super::utils::{parse_file, resolve_input_to_files};

/// A builder for constructing a `FileOnceLoader`.
///
/// it takes a list of glob patterns, resolves them to actual files,
/// and parses the files into `Document`s.
pub struct FileOnceLoaderBuilder {
    paths: Vec<String>,
}

impl FileOnceLoaderBuilder {
    /// Creates a new `FileOnceLoaderBuilder` instance.
    ///
    /// # Arguments
    /// * `paths` - A vector of glob patterns to be loaded.
    pub fn new(paths: Vec<String>) -> Self {
        Self { paths }
    }

    /// Constructs a `FileOnceLoader` instance.
    ///
    /// This method resolves the provided glob patterns, parses the files into
    /// `Document`s, and creates a broadcast channel for the documents.
    ///
    /// # Returns
    /// A `FileOnceLoader` instance.
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

/// A loader that reads documents from files and sends them to subscribers
/// via a broadcast channel.
///
/// Currently can parse PDF files, and treats all other formats as plain text.
pub struct FileOnceLoader {
    tx: broadcast::Sender<Document>,
    documents: Vec<Document>,
    sent: AtomicBool,
}

#[async_trait]
impl Loader for FileOnceLoader {
    /// Subscribes to the loader's broadcast channel to receive documents.
    ///
    /// # Returns
    /// A `tokio::sync::broadcast::Receiver` to receive documents.
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
