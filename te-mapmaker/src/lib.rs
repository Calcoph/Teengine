use std::error::Error;

use pollster;

use te_renderer::initial_config::InitialConfiguration;

mod event_loop;
mod mapmaker;
mod modifiying_instance;

pub fn start_mapmaker(
    config: InitialConfiguration,
    default_model: &str
) -> Result<(), Box<dyn Error>> {
    pollster::block_on(event_loop::run(config, default_model))
}