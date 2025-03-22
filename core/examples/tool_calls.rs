use seedframe::{completion::CompletionError, prelude::*};

#[client(provider = "openai", model = "gpt-4o-mini", tools("add", "subtract"))]
struct CalculatorClient;

/// A function to add two numbers
/// # Arguments
/// * `a`: The first number.
/// * `b`: The second number
#[tool]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// A function to subract two numbers
/// # Arguments
/// * `a`: The first number.
/// * `b`: The second number
#[tool]
fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

#[tokio::main]
async fn main() -> Result<(), CompletionError> {
    let mut c = CalculatorClient::build(
        "You are a calculator, respond with the appropriate tool call".to_string(),
    )
    .await;
    let _resp = c.prompt("What's 7 plus 8").await?;
    let _ = c.run_tools(None).await;
    dbg!(c.export_history());
    Ok(())
}
