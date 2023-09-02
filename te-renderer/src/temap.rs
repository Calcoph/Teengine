use core::panic;
use std::collections::HashMap;

pub struct TeMap {
    pub models: HashMap<String, TeModel>,
    last_model: Option<String>,
}

pub struct TeModel {
    pub offsets: Vec<TeOffset>,
}

pub struct TeOffset {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl TeMap {
    pub fn new() -> Self {
        TeMap {
            models: HashMap::new(),
            last_model: None,
        }
    }

    pub fn from_file(file_name: &str, maps_path: String) -> Self {
        let file_name = file_name.trim_end_matches(".temap");
        let json_temap = std::fs::read_to_string(maps_path + "/" + file_name + ".temap").expect("Could not read file");
        let json_temap = json::parse(&json_temap).expect("Corrupt temap");
        // I know this match is horrible, but I left it as is, because it's also beautiful.
        // Note: no longer a match statement, it was too beautiful. It still is though.
        if let json::JsonValue::Array(models) = json_temap {
            let mut temap = TeMap::new();
            for dict in models {
                if let json::JsonValue::Object(model) = dict {
                    let name = match model.get("name").expect("Invalid .temap") {
                        json::JsonValue::String(n) => n,
                        json::JsonValue::Short(n) => n.as_str(),
                        _ => panic!("Invalid .temap"),
                    };
                    let offsets: Vec<(f32, f32, f32)> = if let json::JsonValue::Array(off) = model.get("offsets").expect("Invalid .temap") {
                        off.iter()
                            .map(|o| if let json::JsonValue::Array(values) = o {
                                (
                                    match values.get(0).expect("Invalid .temap") {
                                        json::JsonValue::Number(n) => f32::from(*n),
                                        _ => panic!("Invalid .temap"),
                                    },
                                    match values.get(1).expect("Invalid .temap") {
                                        json::JsonValue::Number(n) => f32::from(*n),
                                        _ => panic!("Invalid .temap"),
                                    },
                                    match values.get(2).expect("Invalid .temap") {
                                        json::JsonValue::Number(n) => f32::from(*n),
                                        _ => panic!("Invalid .temap"),
                                    },
                                )
                            } else {
                                panic!("Invalid .temap")
                            })
                            .collect()
                    } else {
                        panic!("Invalid .temap")
                    };
                    temap.add_model(name);
                    for offset in offsets {
                        temap.add_instance(offset.0, offset.1, offset.2)
                    }
                } else {
                    panic!("Invalid .temap")
                }
            }
    
            temap
        } else {
            panic!("Invalid .temap")
        }
    }

    pub fn add_model(&mut self, name: &str) {
        if !self.models.contains_key(name) {
            self.models.insert(name.to_string(), TeModel::new());
        }

        self.last_model = Some(name.to_string());
    }

    pub fn add_instance(&mut self, x: f32, y: f32, z: f32) {
        let name = self.last_model.as_ref().expect("No model selected");
        let model = self.models.get_mut(name).expect("Invalid model name");
        model.add_offset(x, y, z);
    }

    pub fn save(&self, file_name: &str, maps_path: String) {
        let contents = self.get_json();
        let file_name = file_name.trim_end_matches(".temap");
        std::fs::write(maps_path + "/" + file_name + ".temap", contents).expect("Could not write to file");
    }

    fn get_json(&self) -> String {
        let indentation = "  ";
        let models = self
            .models
            .iter()
            .map(|(name, model)| {
                String::from(indentation)
                    + "{\n"
                    + indentation
                    + indentation
                    + "\"name\": \""
                    + name
                    + "\",\n"
                    + indentation
                    + indentation
                    + "\"offsets\": "
                    + &model.get_json(indentation)
                    + "\n"
                    + indentation
                    + "},\n"
            })
            .collect::<String>();
        return format!("[\n{}\n]", models.trim_end_matches(",\n"));
    }
}

impl TeModel {
    fn new() -> Self {
        TeModel {
            offsets: Vec::new(),
        }
    }

    fn add_offset(&mut self, x: f32, y: f32, z: f32) {
        self.offsets.push(TeOffset { x, y, z })
    }

    fn get_json(&self, indentation: &str) -> String {
        let offsets = self
            .offsets
            .iter()
            .map(|offset| {
                indentation.to_string() + indentation + &offset.get_json(indentation) + ",\n"
            })
            .collect::<String>();
        return format!(
            "[\n{}\n{}{}]",
            offsets.trim_end_matches(",\n"),
            indentation,
            indentation
        );
    }
}

impl TeOffset {
    fn get_json(&self, indentation: &str) -> String {
        return format!("{}[{},{},{}]", indentation, self.x, self.y, self.z);
    }
}
