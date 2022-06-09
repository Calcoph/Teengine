use std::io::Error;

use pollster;

mod event_loop;
mod consts;
mod gamepad;
mod model;
mod state;
mod texture;
mod resources;
mod camera;
mod mapmaker;
mod temap;

fn main() -> Result<(), Error> {
    pollster::block_on(event_loop::run())
}
