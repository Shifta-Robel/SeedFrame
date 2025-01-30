use std::{io, path::{Path,PathBuf}};
use glob::glob;
use walkdir::WalkDir;
use pdf_extract::extract_text;

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

pub(super) fn parse_file(file_path: &Path) -> io::Result<String> {
    let content = if let Some(ext) = file_path.extension() {
        if ext == "pdf" {
            extract_text(file_path).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("Failed to parse PDF: {e}"))
            })?
        } else {
            std::fs::read_to_string(file_path)?
        }
    } else {
        std::fs::read_to_string(file_path)?
    };

    Ok(content)
}
