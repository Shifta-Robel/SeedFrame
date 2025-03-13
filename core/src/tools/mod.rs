use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args(&self) -> &[ToolArg];

    async fn call(&self, args: Value) -> Result<Value, serde_json::Error>;
    fn output_schema(&self) -> Option<Value> { None }

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

pub struct ToolSet(pub Vec<Box<dyn Tool>>);

#[allow(unused)]
impl ToolSet {
    pub fn find_tool(&self, _name: &str) -> Box<dyn Tool> {
        todo!()
    }
    pub fn add_tool(&mut self, _tool: Box<dyn Tool>) {
        todo!()
    }
    pub fn remove_tool(&mut self, _name: &str) {
        todo!()
    }
    pub fn list_tools(&self) -> Vec<Box<dyn Tool>> {
        todo!()
    }
    pub async fn call(&self, name: &str, args: &str) -> Value {
        todo!()
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
        let schema = schema_for!(T);
        let mut schema_value = serde_json::to_value(&schema).unwrap();

        if let Some(obj) = schema_value.as_object_mut() {
            obj.remove("$schema");
            obj.remove("format");
            obj.remove("title");
            obj.insert("description".to_string(), json!(description));
        }       

        ToolArg {
            name: name.to_string(),
            description: description.to_string(),
            schema: schema_value,
        }
    }
}


/// Represents a tool call requested by the assistant
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
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
