# Seedframe Webscraper

A web scraper integration for [Seedframe](https://github.com/Shifa-Robel/Seedframe).
                                                                                       
This crate provides struct `WebScraper`, which implementing the `seedframe::loader::Loader` trait, that can fetch HTML content from a URL at regular intervals (or once, if no interval is specified) and publishes the results to subscribers. Can optionally be filtered using CSS selectors. The unit of intervals is seconds. The interval and selector fields are optional.

Accepts the following configuration parameters, passed as json to the `config` attribute in the `loader` proc-macro
    - `url`: `String` - url of the page to load content from
    - `interval`: *optional* `u64` - interval at which content gets fetched from the page, happens only once if value is `None`
    - `selector`: *optional* `String` - CSS selector to filter content


```rust
#[loader(
    external = "WebScraper",
    config = r#"{
        "url": "https://example.com",
        "interval": 5,
        "selector": "div.content"
    }"#
)]
struct OurLoader;
```
