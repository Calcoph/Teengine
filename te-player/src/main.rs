use std::io::Error;

use te_mapmaker;

fn main() -> Result<(), Error> {
    te_mapmaker::start_mapmaker(
        (0.0, 20.0, 10.0),
        -90.0,
        -45.0,
        45.0,
        10.0,
        1000.0,
        10.0,
        1.0,
        (32.0, 32.0, 32.0),
        1920,
        1080,
        "ignore/resources".to_string(),
        "ignore/maps".to_string(),
        "ignore/default_texture.png".to_string()
    )
}
