#[derive(Debug, Clone)]
pub struct InitialConfiguration {
    pub tile_size: (f32, f32, f32),
    pub resource_files_directory: String,
    pub map_files_directory: String,
    pub font_dir_path: String,
    pub default_texture_path: String,
    pub icon_path: String,
    pub window_name: String,
    pub camera_position: (f32, f32, f32),
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_fovy: f32,
    pub camera_znear: f32,
    pub camera_zfar: f32,
    pub camera_speed: f32,
    pub camera_sensitivity: f32,
    pub screen_width: u32,
    pub screen_height: u32
}

impl Default for InitialConfiguration {
    fn default() -> Self {
        InitialConfiguration {
            tile_size: (1.0, 1.0, 1.0),
            resource_files_directory: String::from("resources"),
            map_files_directory: String::from("maps"),
            font_dir_path: String::from("resources/font"),
            default_texture_path: String::from("resources/default_texture.png"),
            icon_path: String::from("icon.png"),
            window_name: String::from("Tilengine"),
            camera_position: (0.0, 10.0, 10.0),
            camera_yaw: -90.0,
            camera_pitch: -45.0,
            camera_fovy: 45.0,
            camera_znear: 1.0,
            camera_zfar: 100.0,
            camera_speed: 10.0,
            camera_sensitivity: 0.5,
            screen_width: 1280,
            screen_height: 720
        }
    }
}
