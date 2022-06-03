use std::io::Error;

use pollster;

mod backend;
mod config;
mod gamepad;
mod model;
mod state;
mod texture;
mod resources;
mod camera;
use backend::run;

fn main() -> Result<(), Error> {
    pollster::block_on(run())
}
