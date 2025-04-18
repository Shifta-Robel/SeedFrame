//! A web scraper implementing the `seedframe::loader::Loader` trait.
//!
//! This module provides a `WebScraper` struct that can fetch HTML content from a URL,
//! optionally filter it using CSS selectors, and publish the results at regular intervals.

use seedframe::document::Document;
use seedframe::loader::Loader;
use async_trait::async_trait;
use chrono::Utc;
use scraper::{Html, Selector};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast::{self, Receiver, Sender};
use tokio::sync::Mutex;

/// Configuration structure for the web scraper.
///
/// This is deserialized from the JSON config provided in the `#[loader]` macro.
/// 
/// # Examples
/// 
/// Basic configuration:
/// ```json
/// {
///     "url": "https://example.com",
///     "interval": 5,
///     "selector": "div.content"
/// }
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    url: String,
    interval: Option<u64>,
    selector: Option<String>,
}

/// A web scraper implementation of the `seedframe::Loader` trait.
///
/// The `WebScraper` fetches HTML content from a specified URL at regular intervals
/// (or once, if no interval is specified) and publishes the results to subscribers.
/// It can optionally filter content using CSS selectors.
/// The unit of intervals is seconds. The interval and selector fields are optional.
///
/// # Usage
///
/// Intended for use through the `#[loader]` proc-macro from seedframe:
/// ```ignore
/// #[loader(
///     external = "WebScraper",
///     config = r#"{
///         "url": "https://example.com",
///         "interval": 5,
///         "selector": "div.content"
///     }"#
/// )]
/// struct SomeStruct;
/// ```
pub struct WebScraper {
    sender: Arc<Mutex<Sender<Document>>>,
}

impl WebScraper {
    /// Creates a new `WebScraper` from a JSON configuration string
    /// # Errors
    /// This function will panic if:
    ///  - The provided JSON is malformed and cannot be parsed
    ///  - The JSON contains unknown fields
    pub fn new(json_str: &str) -> Result<Self, serde_json::Error> {
        let config: Config = serde_json::from_str(json_str)?;
        let (sender, _) = broadcast::channel(1);
        let sender = Arc::new(Mutex::new(sender));

        let url = config.url;
        let interval = config.interval.map(Duration::from_secs);
        let selector = config.selector;

        let task_sender = Arc::clone(&sender);
        tokio::spawn(async move {
            let run_once = interval.is_none();
            let selector = selector.and_then(|s| Selector::parse(&s).ok());

            loop {
                match Self::fetch_and_parse(&url, selector.as_ref()).await {
                    Ok(document) => {
                        let sender = task_sender.lock().await;
                        let _ = sender.send(document);
                    }
                    Err(e) => eprintln!("Scraping failed: {e}"),
                }

                if run_once {
                    break;
                }

                if let Some(dur) = interval {
                    tokio::time::sleep(dur).await;
                } else {
                    break;
                }
            }
        });

        Ok(Self { sender })
    }

    /// Fetches and parses website content
    async fn fetch_and_parse(
        url: &str,
        selector: Option<&Selector>,
    ) -> Result<Document, reqwest::Error> {
        let html = reqwest::get(url).await?.text().await?;
        let data = match selector {
            Some(sel) => Html::parse_document(&html)
                .select(sel)
                .map(|e| e.html())
                .collect::<Vec<_>>()
                .join("\n"),
            None => html,
        };

        Ok(Document {
            id: format!("{}-{}", url, Utc::now().timestamp_millis()),
            data,
        })
    }
}

#[async_trait]
impl Loader for WebScraper {
    async fn subscribe(&self) -> Receiver<Document> {
        self.sender.lock().await.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{self, Duration};

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "url": "https://example.com",
            "interval": 60,
            "selector": "div.content"
        }"#;
        
        let config: Result<Config, _> = serde_json::from_str(json);
        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.url, "https://example.com");
        assert_eq!(config.interval, Some(60));
        assert_eq!(config.selector, Some("div.content".to_string()));
    }

    #[tokio::test]
    async fn test_fetch_and_parse_with_selector() {
        let mut mock_server = mockito::Server::new_async().await;
        let url = mock_server.url();
        let mock_server = mock_server.mock("GET", "/")
            .with_status(200)
            .with_body(r#"<html><div class="content">Test</div></html>"#)
            .create();

        let selector = Selector::parse("div.content").unwrap();
        let selector = Some(&selector);
        let result = WebScraper::fetch_and_parse(&url, selector).await;
        
        mock_server.assert();
        assert!(result.is_ok());
        let doc = result.unwrap();
        assert!(doc.data.contains("Test"));
        assert!(!doc.data.contains("html"));
    }

    #[tokio::test]
    async fn test_full_loader_cycle() {
        let mut mock_server = mockito::Server::new_async().await;
        let url = mock_server.url();
        let _ = mock_server.mock("GET", "/")
            .with_body("Test Content")
            .create();

        let json = format!(
            r#"{{
                "url": "{}",
                "interval": 1,
                "selector": null
            }}"#,
            url
        );

        let scraper = WebScraper::new(&json).unwrap();
        let mut receiver = scraper.subscribe().await;

        let first = receiver.recv().await.unwrap();

        let _ = mock_server.mock("GET", "/")
            .with_body("Just Content")
            .create();

        let second = time::timeout(Duration::from_secs(2), receiver.recv())
            .await
            .expect("Didn't receive second message")
            .unwrap();

        assert_ne!(first.id, second.id);
        assert_eq!(first.data, "Test Content");
        assert_eq!(second.data, "Just Content");
    }

    #[tokio::test]
    async fn test_one_time_scraping() {
        let server = mockito::Server::new_async();
        let url = server.await.url();
        let json = &format!("{{\"url\": \"{}\"}}", url);

        let scraper = WebScraper::new(json).unwrap();
        let mut receiver = scraper.subscribe().await;
        
        let mut received = Vec::new();
        while let Ok(doc) = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await {
            received.push(doc.unwrap());
        }

        assert_eq!(received.len(), 1);
        assert!(&received.first().unwrap().id.starts_with(&format!("{url}")));
    }

    #[tokio::test]
    async fn test_invalid_url_handling() {
        let json = r#"{"url": "invalid://url", "interval": null}"#;
        let scraper = WebScraper::new(json).unwrap();
        let mut receiver = scraper.subscribe().await;
        
        let result = time::timeout(Duration::from_secs(1), receiver.recv()).await;
        assert!(result.is_err());
    }
}
