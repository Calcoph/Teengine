use std::io::Error;

use te_renderer::initial_config::InitialConfiguration;
use te_mapmaker;

fn main() -> Result<(), Error> {
    te_mapmaker::start_mapmaker(InitialConfiguration {
        ..Default::default()
    })
}
