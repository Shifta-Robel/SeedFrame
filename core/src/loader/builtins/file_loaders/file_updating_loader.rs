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

pub struct FileUpdatingLoaderBuilder {
    glob_patterns: Vec<String>,
    evaluated_patterns: Vec<Pattern>,
}

impl FileUpdatingLoaderBuilder {
    pub fn new(paths: Vec<String>) -> Result<Self, FileLoaderError> {
        let glob_patterns = paths
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<_, _>>()?;

        Ok(Self {
            glob_patterns: paths,
            evaluated_patterns: glob_patterns,
        })
    }

    pub fn build(self) -> FileUpdatingLoader {
        let files =
            resolve_input_to_files(self.glob_patterns.iter().map(|s| s.as_str()).collect()).unwrap();
        let capacity = if files.len() == 0 {DEFAULT_CHANNEL_CAPACITY} else {files.len()};
        let (tx, _rx) = broadcast::channel(capacity);

        FileUpdatingLoader {
            patterns: self.evaluated_patterns,
            tx,
            sent: AtomicBool::new(false),
        }
    }
}

pub struct FileUpdatingLoader {
    tx: broadcast::Sender<Document>,
    sent: AtomicBool,
    patterns: Vec<Pattern>,
}

#[async_trait]
impl Loader for FileUpdatingLoader {
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
            load_initial(&self.patterns).into_iter().for_each(|doc| {
                self.tx.send(doc).unwrap();
            });

            // setup notify thred
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
                    watcher.watch(path, RecursiveMode::Recursive).unwrap()
                }

                let mut last_event_time = Instant::now();
                let debounce_duration = Duration::from_millis(1000);
                loop {
                    let now = Instant::now();
                    if now.duration_since(last_event_time) >= debounce_duration {
                        for event in evt_rx.iter() {
                            let event = event.unwrap();
                            let out = process_event(&event, &pc);
                            if let None = out {
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

enum EventType {
    Modify,
    Create,
    Delete,
}

fn document_for_event(path: &str, et: EventType) -> Document {
    let file = std::path::Path::new(&path);
    let data = match et {
        EventType::Modify | EventType::Create => parse_file(file).unwrap(),
        EventType::Delete => "".to_string(),
    };
    Document {
        id: path.to_string(),
        data,
    }
}

fn process_event(event: &Event, patterns: &Vec<Pattern>) -> Option<(String, EventType)> {
    let path = event.paths.first()?.to_str()?;
    if !patterns.iter().any(|p| p.matches(path)) {
        return None;
    }

    match event.kind {
        EventKind::Create(CreateKind::File) => Some((path.to_string(), EventType::Create)),
        EventKind::Modify(ModifyKind::Data(_)) => Some((path.to_string(), EventType::Modify)),
        EventKind::Remove(_) => Some((path.to_string(), EventType::Delete)),
        _ => None,
    }
}
