use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::{document::Document, loader::Loader};
use super::utils::{parse_file, resolve_input_to_files};

pub struct FileOnceLoaderBuilder {
    paths: Vec<String>,
}

impl FileOnceLoaderBuilder {
    pub fn new(paths: Vec<String>) -> Self {
        Self { paths }
    }

    pub async fn build(self) -> FileOnceLoader {
        let files = resolve_input_to_files(self.paths.iter().map(|s| s.as_str()).collect()).unwrap();
        let (tx, rx) = broadcast::channel(10);

        for file in files {
            let data = parse_file(&file).unwrap();
            let document = Document {
                id: file.to_string_lossy().to_string(),
                data,
            };
            tx.send(document).unwrap();
        }
        FileOnceLoader{rx}
    }
}

pub struct FileOnceLoader {
    rx: broadcast::Receiver<Document>,
}

#[async_trait]
impl Loader for FileOnceLoader {
    async fn subscribe(&self) -> broadcast::Receiver<Document> {
        self.rx.resubscribe()
    }
}
