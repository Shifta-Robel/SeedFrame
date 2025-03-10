use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::Serialize;
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
                "description": self.description(),
                "parameters": parameters 
            }
        })
    }
}

pub struct ToolSet(pub Vec<Box<dyn Tool>>);

#[allow(unused)]
impl ToolSet {
    fn find_tool(&self, _name: &str) -> Box<dyn Tool> {
        todo!()
    }
    fn add_tool(&mut self, _tool: Box<dyn Tool>) {
        todo!()
    }
    fn remove_tool(&mut self, _name: &str) {
        todo!()
    }
    fn list_tools(&self) -> Vec<Box<dyn Tool>> {
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

pub fn build_parameters_schema(args: &[ToolArg]) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for arg in args {
        properties.insert(arg.name.clone(), arg.schema.clone());
        required.push(json!(arg.name));
    }

    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}
