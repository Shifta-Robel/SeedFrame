use seedframe::{completion::CompletionError, prelude::*};

#[client(provider = "openai", model = "gpt-4o-mini", tools("greet", "capitalize"))]
struct SimpleClient;

/// Greets a user with a personalized message
/// # Arguments
/// * `name`: The name of the person to greet
/// * `mood`: The current mood of the person (happy/tired/excited)
#[tool]
fn greet(name: String, mood: String) -> String {
    format!("Hello {name}! I see you're feeling {mood}.")
}

/// Capitalizes all words in a string
/// # Arguments
/// * `input`: The text to capitalize
#[tool]
fn capitalize(input: String) -> String {
    input.to_uppercase()
}

#[tokio::main]
async fn main() -> Result<(), seedframe::error::Error> {
    let mut client = SimpleClient::build(
        "You are a helpful assistant".to_string(),
    ).await;
    
    client.prompt("Say hello to Rob who's feeling excited")
        .append_tool_response(true)
        .send().await?;

    dbg!(client.export_history());
    Ok(())
}
