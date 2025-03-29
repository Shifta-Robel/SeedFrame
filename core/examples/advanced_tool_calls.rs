use std::fmt::Display;
use schemars::JsonSchema;
use seedframe::prelude::*;
use serde::{Deserialize, Serialize};

#[client(provider = "openai", model = "gpt-4o-mini", tools("schedule_meeting", "convert_temperature"))]
struct AdvancedClient;

/// Meeting configuration parameters
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MeetingConfig {
    /// Duration in minutes
    duration: u32,
    /// Type of meeting
    meeting_type: MeetingType,
    /// Participants email addresses
    participants: Vec<String>
}

/// Type of meeting to schedule
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
enum MeetingType {
    /// Quick sync-up meeting
    Brainstorm,
    /// Detailed planning session
    Planning,
    /// Technical discussion
    TechnicalReview
}

#[tool]
/// Schedules a meeting with complex configuration
/// # Arguments
/// * `config`: The meeting configuration parameters
fn schedule_meeting(config: MeetingConfig) -> String {
    format!("Scheduled {} meeting for {} minutes", config.meeting_type, config.duration)
}

/// Temperature scale specification
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
enum TemperatureScale {
    Celsius,
    Fahrenheit,
    Kelvin
}

/// Temperature conversion result
#[derive(Serialize)]
struct ConversionResult {
    value: f64,
    scale: TemperatureScale,
}

#[tool]
/// Converts temperature between scales
/// # Arguments
/// * `value`: The temperature value to convert
/// * `from`: The scale to convert from
/// * `to`: The scale to convert to
async fn convert_temperature(value: f64, from: TemperatureScale, to: TemperatureScale) -> ConversionResult {
    let converted = match (from, to.clone()) {
        (TemperatureScale::Celsius, TemperatureScale::Fahrenheit) => value * 1.8 + 32.0,
        (TemperatureScale::Fahrenheit, TemperatureScale::Celsius) => (value - 32.0) / 1.8,
        (TemperatureScale::Celsius, TemperatureScale::Kelvin) => value + 273.15,
        (TemperatureScale::Kelvin, TemperatureScale::Celsius) => value - 273.15,
        (TemperatureScale::Fahrenheit, TemperatureScale::Kelvin) => (value - 32.0) / 1.8 + 273.15,
        (TemperatureScale::Kelvin, TemperatureScale::Fahrenheit) => (value - 273.15) * 1.8 + 32.0,
        _ => value,
    };

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    ConversionResult {
        value: converted,
        scale: to,
    }
}

#[tokio::main]
async fn main() -> Result<(), seedframe::error::Error> {
    let mut client = AdvancedClient::build(
        "You are an enterprise assistant".to_string(),
    ).await;

    let response =  client.prompt(
        "Schedule a 90-minute technical review with alice@co.com and bob@co.com"
    ).send().await?;

    println!("Meeting scheduled: {:#?}", response);

    let response = client.prompt(
        "convert the temperature 32.2 from Celcius to Fahrenheit"
    ).send().await?;

    println!("Temprature converted : {:#?}", response);

    Ok(())
}

impl Display for MeetingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Planning => "Planning",
            Self::Brainstorm => "Brainstorm",
            Self::TechnicalReview => "TechnicalReview",
        };
        write!(f, "{s}")
    }
}
