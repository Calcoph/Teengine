use std::io::Error;
use pollster;

mod event_loop;
mod mapmaker;

pub fn start_mapmaker(
    camera_position: (f32, f32, f32),
    camera_yaw: f32,
    camera_pitch: f32,
    camera_fovy: f32,
    camera_znear: f32,
    camera_zfar: f32,
    camera_speed: f32,
    camera_sensitivity: f32,
    tile_size: (f32, f32, f32),
    screen_width: u32,
    screen_height: u32,
    resource_files_directory: String,
    map_files_directory: String,
    default_texture_path: String
) -> Result<(), Error> {
    pollster::block_on(event_loop::run(
        camera_position,
        camera_yaw,
        camera_pitch,
        camera_fovy,
        camera_znear,
        camera_zfar,
        camera_speed,
        camera_sensitivity,
        tile_size,
        screen_width,
        screen_height,
        resource_files_directory,
        map_files_directory,
        default_texture_path
    ))
}