use std::io::Error;
use pollster;

mod event_loop;
mod consts;
mod mapmaker;

fn main() -> Result<(), Error> {
    pollster::block_on(event_loop::run())
}