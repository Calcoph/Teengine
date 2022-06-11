use std::io::Error;
use pollster;

use te_renderer::initial_config::InitialConfiguration;

mod event_loop;
mod mapmaker;

pub fn start_mapmaker(
    config: InitialConfiguration
) -> Result<(), Error> {
    pollster::block_on(event_loop::run(config))
}