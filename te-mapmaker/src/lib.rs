use std::io::Error;
use pollster;

use te_renderer::initial_config::InitialConfiguration;

mod event_loop;
mod mapmaker;
mod modifiying_instance;

pub fn start_mapmaker(
    config: InitialConfiguration,
    default_model: &str
) -> Result<(), Error> {
    pollster::block_on(event_loop::run(config, default_model))
}