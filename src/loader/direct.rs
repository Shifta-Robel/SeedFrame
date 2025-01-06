use async_trait::async_trait;

#[async_trait]
pub trait DirectLoader {
    async fn retrieve(self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    pub struct MyDirectLoader;

    #[async_trait]
    impl DirectLoader for MyDirectLoader {
        async fn retrieve(self) -> String {
            String::from("hello world")
        }
    }

    #[tokio::test]
    async fn test_simple_direct_loader() {
        let my_direct_loader = MyDirectLoader;
        let res = my_direct_loader.retrieve().await;
        assert_eq!(res, String::from("hello world"));
    }
}
