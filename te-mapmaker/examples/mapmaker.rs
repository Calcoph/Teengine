use std::error::Error;

use te_mapmaker::start_mapmaker;
use te_renderer::initial_config::InitialConfiguration;

fn main() -> Result<(), Box<dyn Error>> {
    start_mapmaker(
        InitialConfiguration {
            ..Default::default()
        },
        "box02.glb",
    )
}
