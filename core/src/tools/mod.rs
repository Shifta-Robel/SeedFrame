use async_trait::async_trait;
use schemars::{gen::SchemaSettings, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args(&self) -> &[ToolArg];

    async fn call(&self, args: &str) -> Result<Value, ToolError>;
    fn output_schema(&self) -> Option<Value> {
        None
    }

    fn default_serializer(&self) -> Value {
        let parameters = build_parameters_schema(self.args());
        json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "strict": true,
                "description": self.description(),
                "parameters": parameters
            }
        })
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("Faild to execute the call")]
    ToolCallError(#[from] Box<dyn std::error::Error + Send + Sync>),
    #[error("Json Error")]
    JsonError(#[from] serde_json::Error),
}

pub enum ExecutionStrategy {
    FailEarly,
    BestEffort,
}

pub struct ToolSet(pub Vec<Box<dyn Tool>>, pub ExecutionStrategy);

#[derive(Debug, Error)]
pub enum ToolSetError {
    #[error("Failed to find tool")]
    ToolNotFound,
    #[error("Client message history is empty")]
    EmptyMessageHistory,
    #[error("Last entry in client message history doesn't contain a ToolCall")]
    LastMessageNotAToolCall,
    #[error("Tool error: ")]
    ToolError(#[from] ToolError),
}

#[allow(unused)]
impl ToolSet {
    pub fn find_tool(&self, name: &str) -> Result<&Box<dyn Tool>, ToolSetError> {
        self.0
            .iter()
            .find(|t| t.name() == name)
            .ok_or(ToolSetError::ToolNotFound)
    }

    pub fn add_tool(&mut self, tool: Box<dyn Tool>) {
        self.0.push(tool)
    }

    pub fn remove_tool(&mut self, name: &str) -> Result<(), ToolSetError> {
        let pos = self
            .0
            .iter()
            .position(|t| t.name() == name)
            .ok_or(ToolSetError::ToolNotFound)?;
        self.0.remove(pos);
        Ok(())
    }

    pub fn list_tools(&self) -> Vec<Box<dyn Tool>> {
        todo!()
    }

    pub async fn call(
        &self,
        id: &str,
        name: &str,
        args: &str,
    ) -> Result<ToolResponse, ToolSetError> {
        let tool = self.find_tool(name)?;
        let v = tool.call(args).await.map_err(ToolSetError::from)?;
        Ok(ToolResponse {
            id: id.to_owned(),
            name: name.to_owned(),
            content: v,
        })
    }
}

#[allow(unused)]
pub struct ToolArg {
    name: String,
    description: String,
    schema: Value,
}

impl ToolArg {
    pub fn new<T: JsonSchema + Serialize>(name: &str, description: &str) -> Self {
        let settings = SchemaSettings::default().with(|s| {
            s.inline_subschemas = true;
        });
        let generator = settings.into_generator();
        let schema = generator.into_root_schema_for::<T>();
        let mut schema_value = serde_json::to_value(&schema).unwrap();

        if let Some(obj) = schema_value.as_object_mut() {
            obj.remove("$schema");
            obj.remove("format");
            obj.remove("title");
            obj.insert("description".to_string(), json!(description));
        }
        process_json_value(&mut schema_value);

        ToolArg {
            name: name.to_string(),
            description: description.to_string(),
            schema: schema_value,
        }
    }
}

fn process_json_value(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            let fields_to_remove = ["$schema", "format", "title", "minimum"];
            fields_to_remove.iter().for_each(|&f| {
                if obj.get(f).map_or(false, |v| v.is_string() || v.is_number()) {
                    obj.remove(f);
                }
            });
            if let Some(v) = obj.get("oneOf").cloned() {
                obj.remove("oneOf");
                obj.insert("anyOf".to_string(), v);
            };

            if obj.contains_key("properties") {
                obj.insert("additionalProperties".to_string(), json!(false));
            }
            for (_, v) in obj.iter_mut() {
                process_json_value(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for elem in arr.iter_mut() {
                process_json_value(elem);
            }
        }
        _ => {}
    }
}

/// Represents a tool call requested by the assistant
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Represents the output of a tool execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResponse {
    pub id: String,
    pub name: String,
    pub content: serde_json::Value,
}

pub fn build_parameters_schema(args: &[ToolArg]) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for arg in args {
        let mut schema = arg.schema.clone();
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("minimum");
        }
        properties.insert(arg.name.clone(), schema.clone());
        required.push(json!(arg.name));
    }

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}
