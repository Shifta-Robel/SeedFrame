use async_trait::async_trait;
use glob::Pattern;
use tokio::sync::broadcast;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{document::Document, loader::Loader};
use super::{utils::{load_initial, resolve_input_to_files}, FileLoaderError};

/// A builder for constructing a `FileOnceLoader`.
///
/// it takes a list of glob patterns, resolves them to actual files,
/// and parses the files into `Document`s.
pub struct FileOnceLoaderBuilder {
    glob_patterns: Vec<String>,
    evaluated: Vec<glob::Pattern>
}

impl FileOnceLoaderBuilder {
    /// Creates a new `FileOnceLoaderBuilder` instance.
    ///
    /// # Arguments
    /// * `glob_patterns` - A vector of glob pattern strings to be loaded.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `FileOnceLoaderBuilder` instance.
    /// * `Err(FileLoaderError)` - An error if initialization fails.
    pub fn new(glob_patterns: Vec<String>) -> Result<Self, FileLoaderError> {
        let evaluated = glob_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<_, _>>()?;

        Ok(Self { glob_patterns, evaluated })
    }

    /// Constructs a `FileOnceLoader` instance.
    ///
    /// This method resolves the glob patterns, parses the files into
    /// `Document`s, and creates a broadcast channel for the documents.
    ///
    /// # Returns
    /// A `FileOnceLoader` instance.
    pub fn build(self) -> FileOnceLoader {
        let files = resolve_input_to_files(self.paths.iter().map(|s| s.as_str()).collect()).unwrap();
        let (tx, _rx) = broadcast::channel(files.len());

        let documents = load_initial(&self.patterns);

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
    /// A `tokio::sync::broadcast::Receiver<Document>`.
    async fn subscribe(&self) -> broadcast::Receiver<Document> {
        let receiver = self.tx.subscribe();
        if !self.sent.load(Ordering::Acquire) &&
            self.sent.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                for doc in &self.documents {
                    self.tx.send(doc.clone()).unwrap();
                }
        }
        receiver
    }
}
