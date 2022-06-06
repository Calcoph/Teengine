use std::io::Error;

use pollster;

mod event_loop;
mod config;
mod gamepad;
mod model;
mod state;
mod texture;
mod resources;
mod camera;
mod mapmaker;

fn main() -> Result<(), Error> {
    pollster::block_on(event_loop::run())
}
