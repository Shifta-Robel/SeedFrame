use glob::{glob, Pattern};
#[cfg(feature = "pdf")]
use pdf_extract::extract_text;
#[cfg(feature = "pdf")]
use tracing::error;
use std::{
    io,
    path::{Path, PathBuf},
};
use tracing::{info, instrument};
use walkdir::WalkDir;

use crate::document::Document;

/// Resolves a list of glob patterns into a list of file paths.
///
/// # Arguments
/// * `inputs` - glob patterns as a vector of `&str`s.
///
/// # Returns
/// * `Ok(Vec<PathBuf>)` - A vector of resolved file paths.
/// * `Err(io::Error)`
pub(super) fn resolve_input_to_files(inputs: Vec<&str>) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for input in inputs {
        for entry in glob(input).unwrap() {
            match entry {
                Ok(path) => {
                    if path.is_dir() {
                        for entry in WalkDir::new(path) {
                            let entry = entry?;
                            if entry.file_type().is_file() {
                                files.push(entry.path().to_path_buf());
                            }
                        }
                    } else if path.is_file() {
                        files.push(path);
                    }
                }
                Err(e) => eprintln!("Glob error: {e}"),
            }
        }
    }

    Ok(files)
}

#[instrument]
/// Parses the content of a file based on its extension.
///
/// This function reads the content of a file. If the file is a PDF, it uses the `pdf_extract` crate
/// to extract text from the PDF. For all other file types, it reads the file as plain text.
///
/// # Arguments
/// * `file_path` - The path to the file to parse.
///
/// # Returns
/// * `Ok(String)` - The content of the file as a string.
/// * `Err(io::Error)` - An error if the file cannot be read or parsed.
pub(super) fn parse_file(file_path: &Path) -> io::Result<String> {
    let content = std::fs::read_to_string(file_path)?;
    info!("Successfully parsed file: {:?}", file_path);
    Ok(content)
}

#[instrument]
/// Parses the content of a file based on its extension.
///
/// This function reads the content of a file. If the file is a PDF, it uses the `pdf_extract` crate
/// to extract text from the PDF. For all other file types, it reads the file as plain text.
///
/// # Arguments
/// * `file_path` - The path to the file to parse.
///
/// # Returns
/// * `Ok(String)` - The content of the file as a string.
/// * `Err(io::Error)` - An error if the file cannot be read or parsed.
#[cfg(feature = "pdf")]
pub(super) fn parse_file(file_path: &Path) -> io::Result<String> {
    let content = if let Some(ext) = file_path.extension() {
        if ext == "pdf" {
            extract_text(file_path).map_err(|e| {
                error!("Failed to parse PDF: {e}");
                io::Error::new(io::ErrorKind::Other, format!("Failed to parse PDF: {e}"))
            })?
        } else {
            std::fs::read_to_string(file_path)?
        }
    } else {
        std::fs::read_to_string(file_path)?
    };
    info!("Successfully parsed file: {:?}", file_path);
    Ok(content)
}

pub(super) fn load_initial(patterns: &[Pattern]) -> Vec<Document> {
    let files = resolve_input_to_files(patterns.iter().map(|s| s.as_str()).collect()).unwrap();
    let mut documents: Vec<Document> = vec![];
    for file in files {
        let data = parse_file(&file).unwrap();
        let document = Document {
            id: file.to_string_lossy().to_string(),
            data,
        };
        info!("Successfully loaded document: {:?}", document.id.clone());
        documents.push(document);
    }
    documents
}

pub(super) fn extract_parent_dir(pattern: &str) -> PathBuf {
    let glob_indicators = ['*', '?', '[', '!'];
    let dirs: Vec<_> = pattern.split('/').collect();
    let idx = dirs
        .iter()
        .position(|part| glob_indicators.iter().any(|&c| part.contains(c)))
        .unwrap_or(dirs.len());

    let base_path = dirs[..idx].join("/");
    let path = Path::new(&base_path);
    let path = if path.is_file() {
        path.parent().unwrap()
    } else {
        path
    };

    std::fs::canonicalize(path).unwrap()
}

pub(super) fn get_dirs_to_watch(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths
        .iter()
        .filter(|path| {
            !paths
                .iter()
                .any(|other| *path != other && path.starts_with(other))
        })
        .cloned()
        .collect()
}
