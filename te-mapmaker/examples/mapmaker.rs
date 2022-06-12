use std::io::Error;

use te_mapmaker::start_mapmaker;
use te_renderer::initial_config::InitialConfiguration;

fn main() -> Result<(), Error> {
    start_mapmaker(InitialConfiguration {
        ..Default::default()
    })
}