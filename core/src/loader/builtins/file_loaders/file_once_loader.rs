use async_trait::async_trait;
use glob::Pattern;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::broadcast;
use tracing::{debug, error, info, instrument};

use super::{utils::load_initial, FileLoaderError};
use crate::{document::Document, loader::Loader};

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
/// A builder for constructing a `FileOnceLoader`.
///
/// it takes a list of glob patterns, resolves them to actual files,
/// and parses the files into `Document`s.
pub struct FileOnceLoaderBuilder {
    glob_patterns: Vec<String>,
    evaluated: Vec<glob::Pattern>,
}

impl FileOnceLoaderBuilder {
    #[instrument]
    /// Creates a new `FileOnceLoaderBuilder` instance.
    ///
    /// # Arguments
    /// * `glob_patterns` - A vector of glob pattern strings to be loaded.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `FileOnceLoaderBuilder` instance.
    /// * `Err(FileLoaderError)` - An error if initialization fails.
    pub fn new(glob_patterns: Vec<String>) -> Result<Self, FileLoaderError> {
        let evaluated: Vec<Pattern> = glob_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<_, _>>()?;
        info!("Successfully evaluated {} glob patterns", evaluated.len());

        Ok(Self {
            glob_patterns,
            evaluated,
        })
    }

    #[instrument(fields(self = format!("FileOnceLoaderBuilder {{glob_patterns: {:?}}}", self.glob_patterns)))]
    /// Constructs a `FileOnceLoader` instance.
    ///
    /// This method resolves the glob patterns, parses the files into
    /// `Document`s, and creates a broadcast channel for the documents.
    ///
    /// # Returns
    /// * `Ok(FileOnceLoader)` - A new `FileOnceLoader` instance.
    /// * `Err(FileLoaderError)` - An error if build fails.
    pub fn build(self) -> Result<FileOnceLoader, FileLoaderError> {
        let documents = load_initial(&self.evaluated);
        if documents.is_empty() {
            error!("No documents matched the provided glob patterns");
            Err(FileLoaderError::NoMatchingDocuments)?;
        };
        let (tx, _rx) = broadcast::channel(documents.len());
        debug!(
            "broadcast channel with capacity: {} created",
            documents.len()
        );

        Ok(FileOnceLoader {
            tx,
            documents,
            sent: AtomicBool::new(false),
        })
    }
}

#[derive(Debug)]
/// A loader that reads documents from files and sends them to subscribers
/// via a broadcast channel.
///
/// Currently can parse PDF files, and treats all other formats as plain text.
pub struct FileOnceLoader {
    tx: broadcast::Sender<Document>,
    documents: Vec<Document>,
    sent: AtomicBool,
}

impl FileOnceLoader {
    /// Here for making use by the `proc_macro` lib more convinient, just calls
    /// `FileOnceLoaderBuilder::new()`
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::missing_errors_doc)]
    pub fn new(glob_patterns: Vec<String>) -> Result<FileOnceLoaderBuilder, FileLoaderError> {
        FileOnceLoaderBuilder::new(glob_patterns)
    }
}

#[async_trait]
impl Loader for FileOnceLoader {
    #[instrument(fields(self = format!("FileOnceLoader {{sent: {}}}", self.sent.load(Ordering::Acquire))))]
    /// Subscribes to the loader's broadcast channel to receive documents.
    ///
    /// # Returns
    /// A `tokio::sync::broadcast::Receiver<Document>`.
    async fn subscribe(&self) -> broadcast::Receiver<Document> {
        let receiver = self.tx.subscribe();
        let mut sent_docs_count = 0;
        if !self.sent.load(Ordering::Acquire)
            && self
                .sent
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            for doc in &self.documents {
                if let Err(e) = self.tx.send(doc.clone()) {
                    error!("Loader failed to send document: {} to subscribers", e.0.id);
                } else {
                    sent_docs_count += 1;
                }
            }
            info!(
                "Loader sent {} of {} documents to subscribers",
                self.documents.len(),
                sent_docs_count
            );
        }
        receiver
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::time::{timeout, Duration};

    async fn create_test_files(dir: &std::path::Path, names: &[&str]) {
        for name in names {
            tokio::fs::write(dir.join(name), "test content")
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_invalid_glob_patterns() {
        let invalid_patterns = vec![
            "*",
            "?",
            "[",
            "]",
            "{",
            "}",
            "!",
            "*.*",
            "*.txt*",
            "*.txt?",
            "*.txt[a-z]",
            "*.txt{a,b}",
            "*.txt!",
            "*.txt,*.pdf",
            "*.txt *.pdf",
        ]
        .iter()
        .map(|p| (*p).to_string())
        .collect::<Vec<String>>();
        let result = FileOnceLoaderBuilder::new(invalid_patterns);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_loads_exact_files() {
        let dir = tempdir().unwrap();
        let file_names = ["t1.txt", "t2.txt"];
        create_test_files(dir.path(), &file_names).await;

        let paths = file_names
            .iter()
            .map(|name| dir.path().join(name).to_str().unwrap().to_string())
            .collect();

        let loader = FileOnceLoaderBuilder::new(paths).unwrap().build().unwrap();
        let mut receiver = loader.subscribe().await;

        let mut received = Vec::new();
        while let Ok(doc) = timeout(Duration::from_millis(100), receiver.recv()).await {
            received.push(doc.unwrap());
        }

        assert_eq!(received.len(), 2);
    }

    #[tokio::test]
    async fn test_glob_pattern_matching() {
        let dir = tempdir().unwrap();
        create_test_files(dir.path(), &["t1.txt", "t2.txt", "img.jpg"]).await;

        let glob_path = dir.path().join("*.txt").to_str().unwrap().to_string();
        let loader = FileOnceLoaderBuilder::new(vec![glob_path])
            .unwrap()
            .build()
            .unwrap();

        let mut receiver = loader.subscribe().await;
        let mut received = Vec::new();
        while let Ok(doc) = timeout(Duration::from_millis(100), receiver.recv()).await {
            received.push(doc.unwrap());
        }

        assert_eq!(received.len(), 2);
    }

    #[tokio::test]
    async fn test_no_matching_files() {
        let dir = tempdir().unwrap();
        let glob_path = dir.path().join("*.md").to_str().unwrap().to_string();

        let loader = FileOnceLoaderBuilder::new(vec![glob_path]).unwrap().build();
        assert!(loader.is_err());
    }
}
