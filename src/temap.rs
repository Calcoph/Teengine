use core::panic;
use std::collections::HashMap;

pub struct TeMap {
    pub models: HashMap<String, TeModel>,
    last_model: Option<String>
}

pub struct TeModel {
    pub offsets: Vec<TeOffset>
}

pub struct TeOffset {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl TeMap {
    pub fn new() -> Self {
        TeMap {
            models: HashMap::new(),
            last_model: None
        }
    }

    pub fn from_file(file_name: &str) -> Self {
        let file_name = file_name.trim_end_matches(".temap");
        let json_temap = std::fs::read_to_string("ignore/maps/".to_string() + file_name + ".temap").unwrap();
        let json_temap = json::parse(&json_temap).unwrap();
        // I know this match is horrible, but I left it as is, because it's also beautiful
        match json_temap {
            json::JsonValue::Array(models) => {
                let mut temap = TeMap::new();
                for dict in models {
                    match dict {
                        json::JsonValue::Object(model) => {
                            let name = match model.get("name").expect("Invalid .temap") {
                                json::JsonValue::String(n) => n,
                                json::JsonValue::Short(n) => n.as_str(),
                                _ => panic!("Invalid .temap")
                            };
                            let offsets: Vec<(f32, f32, f32)> = match model.get("offsets").expect("Invalid .temap") {
                                json::JsonValue::Array(off) => {
                                    off.iter().map(|o| {
                                        match o {
                                            json::JsonValue::Array(values) => {
                                                (
                                                    match values.get(0).expect("Invalid .temap") {
                                                        json::JsonValue::Number(n) => f32::from(*n),
                                                        _ => panic!("Invalid .temap")
                                                    },
                                                    match values.get(1).expect("Invalid .temap") {
                                                        json::JsonValue::Number(n) => f32::from(*n),
                                                        _ => panic!("Invalid .temap")
                                                    },
                                                    match values.get(2).expect("Invalid .temap") {
                                                        json::JsonValue::Number(n) => f32::from(*n),
                                                        _ => panic!("Invalid .temap")
                                                    },
                                                )
                                            },
                                            _ => panic!("Invalid .temap")
                                        }
                                    }).collect()
                                },
                                _ => panic!("Invalid .temap")
                            };
                            temap.add_model(name);
                            for offset in offsets {
                                temap.add_instance(offset.0, offset.1, offset.2)
                            }
                        },
                        _ => panic!("Invalid .temap")
                    }
                };

                temap
            },
            _ => panic!("Invalid .temap")
        }
    }

    pub fn add_model(&mut self, name: &str) {
        if !self.models.contains_key(name) {
            self.models.insert(name.to_string(), TeModel::new());
        }

        self.last_model = Some(name.to_string());
    }

    pub fn add_instance(&mut self, x: f32, y: f32, z: f32) {
        let name = self.last_model.as_ref().unwrap();
        let model = self.models.get_mut(name).unwrap();
        model.add_offset(x, y, z);
    }

    pub fn save(&self, file_name: &str) {
        let contents = self.get_json();
        let file_name = file_name.trim_end_matches(".temap");
        std::fs::write("ignore/maps/".to_string() + file_name + ".temap", contents).unwrap();
    }

    fn get_json(&self) -> String {
        let indentation = "  ";
        let models = self.models.iter()
            .map(|(name, model)| {
                String::from(indentation) + "{\n"
                + indentation + indentation + "\"name\": \"" + name + "\",\n"
                + indentation + indentation + "\"offsets\": " + &model.get_json(indentation) + "\n"
                + indentation + "},\n"
            }).collect::<String>();
        return format!("[\n{}\n]", models.trim_end_matches(",\n"));
    }
}

impl TeModel {
    fn new() -> Self {
        TeModel { offsets: Vec::new() }
    }

    fn add_offset(&mut self, x: f32, y: f32, z: f32) {
        self.offsets.push(TeOffset {
            x,
            y,
            z
        })
    }
    
    fn get_json(&self, indentation: &str) -> String {
        let offsets = self.offsets.iter()
            .map(|offset| indentation.to_string() + indentation + &offset.get_json(indentation) + ",\n")
            .collect::<String>();
        return format!("[\n{}\n{}{}]", offsets.trim_end_matches(",\n"), indentation, indentation);
    }
}

impl TeOffset {
    fn get_json(&self, indentation: &str) -> String {
        return format!("{}[{},{},{}]", indentation, self.x, self.y, self.z);
    }
}