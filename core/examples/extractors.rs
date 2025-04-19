use schemars::JsonSchema;
use seedframe::prelude::*;
use seedframe::providers::completions::OpenAI;
use serde::Deserialize;

#[client(provider = "OpenAI")]
struct ExtractorClient;

#[allow(unused)]
#[derive(JsonSchema, Deserialize, Extractor, Debug)]
struct Person {
    /// Full name of the person
    name: String,
    age: u8,
    #[schemars(required)]
    /// Optional: Person's job title
    job_title: Option<String>,
    /// List of hobbies
    hobbies: Vec<String>,
}

#[allow(unused)]
#[derive(JsonSchema, Deserialize, Extractor, Debug)]
struct MeetingDetails {
    /// Date of the meeting in YYYY-MM-DD format
    date: String,
    /// Time of the meeting in HH:MM format
    time: String,
    /// Purpose of the meeting
    purpose: String,
    /// List of attendees
    attendees: Vec<Person>,
}

#[tokio::main]
async fn main() -> Result<(), seedframe::error::Error> {
    let mut client = ExtractorClient::build("You are a helpful assistant").await;

    let person_text = "My colleague John Doe is 28 years old. He works as a software engineer and enjoys hiking, reading, and playing chess, his email is johhnydoep@mail.com";
    let person = client.prompt(person_text).extract::<Person>().await?;
    println!("Extracted Person:\n{person:#?}\n");

    let meeting_text = "We have a team meeting scheduled for 2026-2-15 at 11:30. \
                       Purpose is quarterly planning. Attendees include: \
                       John Doe (35, Product Manager, hobbies: golf), \
                       Alice Smith (29, hobbies: painting, yoga), \
                       and Bob Johnson (42, CTO).";
    let meeting = client
        .prompt(meeting_text)
        .extract::<MeetingDetails>()
        .await?;
    println!("Extracted Meeting:\n{meeting:#?}\n");

    Ok(())
}
