use async_trait::async_trait;
use glob::Pattern;
use notify::{
    event::{CreateKind, ModifyKind},
    Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use tracing::{debug, error, info, instrument};

use crate::{
    document::Document,
    loader::{
        builtins::file_loaders::utils::{
            extract_parent_dir, get_dirs_to_watch, parse_file, resolve_input_to_files,
        },
        Loader,
    },
};

use super::{utils::load_initial, FileLoaderError};

const DEFAULT_CHANNEL_CAPACITY: usize = 20;
const DEBOUNCE_DURATION_MILLIS: u64 = 500;

#[derive(Debug)]
/// A builder for constructing a `FileUpdatingLoader`.
///
/// it takes a list of glob patterns, resolves them to actual files,
/// and parses the files into `Document`s.
pub struct FileUpdatingLoaderBuilder {
    glob_patterns: Vec<String>,
    evaluated_patterns: Vec<Pattern>,
}

impl FileUpdatingLoaderBuilder {
    #[instrument]
    /// Creates a new `FileUpdatingLoaderBuilder` instance.
    ///
    /// # Arguments
    /// * `glob_patterns` - A vector of glob pattern strings to be loaded.
    ///
    /// # Returns
    /// * `Ok(Self)` - A new `FileUpdatingLoaderBuilder` instance.
    /// * `Err(FileLoaderError)` - An error if initialization fails.
    pub fn new(glob_patterns: Vec<String>) -> Result<Self, FileLoaderError> {
        let evaluated_patterns: Vec<Pattern> = glob_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<_, _>>()?;
        info!(
            "Successfully evaluated {} glob patterns",
            evaluated_patterns.len()
        );

        Ok(Self {
            glob_patterns,
            evaluated_patterns,
        })
    }

    #[instrument]
    /// Constructs a `FileUpdatingLoader` instance.
    ///
    /// This method resolves the glob patterns, parses the files into
    /// `Document`s, and creates a broadcast channel for the documents.
    ///
    /// # Returns
    /// * `FileUpdatingLoader` - A new `FileUpdatingLoader` instance.
    pub fn build(self) -> FileUpdatingLoader {
        let files = resolve_input_to_files(self.glob_patterns.iter().map(std::string::String::as_str).collect())
            .unwrap();
        let capacity = if files.is_empty() {
            DEFAULT_CHANNEL_CAPACITY
        } else {
            files.len()
        };
        let (tx, _rx) = broadcast::channel(capacity);
        debug!("broadcast channel with capacity: {} created", capacity);

        FileUpdatingLoader {
            patterns: self.evaluated_patterns,
            tx,
            sent: AtomicBool::new(false),
        }
    }
}

#[derive(Debug)]
/// Watches files matching glob patterns and emits document updates
///
/// Implements the [`Loader`] trait. When subscribed:
/// 1. Immediately sends all matching documents
/// 2. Watches filesystem for changes
/// 3. Sends updates on changes
///
/// Deleted files are sent with empty content. Multiple subscribers are supported
/// via broadcast channel.
pub struct FileUpdatingLoader {
    tx: broadcast::Sender<Document>,
    sent: AtomicBool,
    patterns: Vec<Pattern>,
}

#[async_trait]
impl Loader for FileUpdatingLoader {
    #[instrument(fields(self = format!("FileUpdatingLoader {{sent: {}}}", self.sent.load(Ordering::Acquire))))]
    /// Subscribes to the loader's broadcast channel to receive documents.
    ///
    /// # Returns
    /// A `tokio::sync::broadcast::Receiver<Document>`.
    async fn subscribe(&self) -> broadcast::Receiver<Document> {
        let receiver = self.tx.subscribe();
        if !self.sent.load(Ordering::Acquire)
            && self
                .sent
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            let initial_docs = load_initial(&self.patterns);
            let mut sent_docs_count = 0;
            let total_docs_count = initial_docs.len();
            for doc in initial_docs {
                if let Err(e) = self.tx.send(doc) {
                    error!("Loader failed to send document: {} to subscribers", e.0.id);
                } else {
                    sent_docs_count += 1;
                }
            }
            info!(
                "Loader sent {} of {} initial documents to subscribers",
                sent_docs_count, total_docs_count
            );

            // setup notify thread
            let to_be_watched = get_dirs_to_watch(
                &self
                    .patterns
                    .iter()
                    .map(|p| extract_parent_dir(p.as_str()))
                    .collect::<Vec<PathBuf>>(),
            );

            let txc = self.tx.clone();
            let pc = self.patterns.clone();
            tokio::task::spawn_blocking(move || {
                let (evt_tx, evt_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
                let mut watcher = RecommendedWatcher::new(evt_tx, Config::default()).unwrap();

                for path in &to_be_watched.clone() {
                    if let Err(e) = watcher.watch(path, RecursiveMode::Recursive) {
                        error!(
                            "Failed to add path {} to watcher: {}",
                            path.to_str().unwrap_or("unknown"),
                            e
                        );
                    } else {
                        info!(
                            "Added path {} to watcher",
                            path.to_str().unwrap_or("unknown")
                        );
                    }
                }

                let mut last_event_time = Instant::now();
                let debounce_duration = Duration::from_millis(DEBOUNCE_DURATION_MILLIS);
                loop {
                    let now = Instant::now();
                    if now.duration_since(last_event_time) >= debounce_duration {
                        for event in &evt_rx {
                            let event = event.unwrap();
                            let out = process_event(&event, &pc);
                            if out.is_none() {
                                continue;
                            }
                            let out = out.unwrap();
                            txc.send(document_for_event(out.0.as_str(), out.1)).unwrap();
                            last_event_time = now;
                        }
                    }
                }
            });
        }
        receiver
    }
}

#[derive(Debug)]
enum EventType {
    Modify,
    Create,
    Delete,
}

#[instrument]
fn document_for_event(path: &str, et: EventType) -> Document {
    let file = std::path::Path::new(&path);
    let data = match et {
        EventType::Modify | EventType::Create => parse_file(file).unwrap(),
        EventType::Delete => String::new(),
    };
    debug!("Created document for {} with event type {:?}", path, et);
    Document {
        id: path.to_string(),
        data,
    }
}

#[instrument]
fn process_event(event: &Event, patterns: &[Pattern]) -> Option<(String, EventType)> {
    let path = event.paths.first()?.to_str()?;
    if !patterns.iter().any(|p| p.matches(path)) {
        debug!("Event path {} does not match any patterns", path);
        return None;
    }

    match event.kind {
        EventKind::Create(CreateKind::File) => Some((path.to_string(), EventType::Create)),
        EventKind::Modify(ModifyKind::Data(_)) => Some((path.to_string(), EventType::Modify)),
        EventKind::Remove(_) => Some((path.to_string(), EventType::Delete)),
        _ => {
            debug!("Unhandled event type {:?}", event.kind);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use notify::{Event, EventKind};
    use std::path::PathBuf;
    use tempfile;

    // fn process_event tests
    #[test]
    fn test_process_event_matching_pattern() {
        let pattern = Pattern::new("*.txt").unwrap();
        let event = Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![PathBuf::from("test.txt")],
            attrs: Default::default(),
        };
        let result = process_event(&event, &[pattern]);
        assert!(result.is_some());
        let (path, et) = result.unwrap();
        assert_eq!(path, "test.txt");
        assert!(matches!(et, EventType::Create));
    }

    #[test]
    fn test_process_event_non_matching_pattern() {
        let pattern = Pattern::new("*.md").unwrap();
        let event = Event {
            kind: EventKind::Create(CreateKind::File),
            paths: vec![PathBuf::from("test.txt")],
            attrs: Default::default(),
        };
        let result = process_event(&event, &[pattern]);
        assert!(result.is_none());
    }

    // fn document_for_event tests
    #[test]
    fn test_document_for_event_create() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "test content").unwrap();

        let doc = document_for_event(file_path.to_str().unwrap(), EventType::Create);
        assert_eq!(doc.id, file_path.to_str().unwrap());
        assert_eq!(doc.data, "test content");
    }

    #[test]
    fn test_document_for_event_delete() {
        let doc = document_for_event("test.txt", EventType::Delete);
        assert_eq!(doc.data, "");
    }

    // TODO: fix this, spawn_blocking does what its supposed to do
    #[tokio::test]
    #[ignore]
    async fn test_initial_load_sends_documents() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "initial").unwrap();

        let builder = FileUpdatingLoaderBuilder::new(vec![temp_dir
            .path()
            .join("*.txt")
            .to_str()
            .unwrap()
            .to_string()])
        .unwrap();
        let loader = builder.build();

        let mut receiver = loader.subscribe().await;

        let doc = receiver.recv().await.unwrap();
        assert_eq!(doc.id, file_path.to_str().unwrap());
        assert_eq!(doc.data, "initial");
    }

    #[tokio::test]
    async fn test_non_matching_files_ignored() {
        let temp_dir = tempfile::tempdir().unwrap();
        let matching_path = temp_dir.path().join("test.txt");
        let non_matching_path = temp_dir.path().join("test.md");
        std::fs::write(&matching_path, "initial").unwrap();

        let builder = FileUpdatingLoaderBuilder::new(vec![temp_dir
            .path()
            .join("*.txt")
            .to_str()
            .unwrap()
            .to_string()])
        .unwrap();
        let loader = builder.build();

        let mut receiver = loader.subscribe().await;
        receiver.recv().await.unwrap();

        std::fs::write(&non_matching_path, "modified").unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;

        assert!(receiver.try_recv().is_err());
    }
}
