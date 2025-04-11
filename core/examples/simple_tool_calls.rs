use seedframe::prelude::*;

#[client(
    provider = "openai",
    model = "gpt-4o-mini",
    tools("capitalize", "greet")
)]
struct SimpleClient;

/// Greets a user with a personalized message
/// # Arguments
/// * `name`: The name of the person to greet
/// * `mood`: The current mood of the person (happy/tired/excited)
#[tool]
fn greet(name: String, mood: String) {
    println!("Hello {name}! I see you're feeling {mood}.");
}

/// Capitalizes all words in a string
/// # Arguments
/// * `input`: The text to capitalize
#[tool]
fn capitalize(input: String, State(state): State<AppState>) -> String {
    format!(
        "capitalize: {}, state: {}",
        input.to_uppercase(),
        state.some_number,
    )
}

#[derive(Debug)]
struct AppState {
    some_number: i32,
}

#[tokio::main]
async fn main() -> Result<(), seedframe::error::Error> {
    let mut client = SimpleClient::build("You are a helpful assistant")
        .await
        .with_state(AppState { some_number: 3 })?;

    client
        .prompt("Capitalize the word 'capital'")
        .append_tool_response(true)
        .send()
        .await?;

    dbg!(client.export_history());
    Ok(())
}
