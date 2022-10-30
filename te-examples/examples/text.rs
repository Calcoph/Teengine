use te_renderer::{initial_config::InitialConfiguration, state::TeColor};

fn main() {
    pollster::block_on(as_main());
}

async fn as_main() {
    let (
        event_loop,
        gpu,
        window,
        te_state,
    ) = te_player::prepare(InitialConfiguration {
        screen_width: 1000,
        screen_height: 500,
        ..InitialConfiguration::default()
    }, true).await.expect("Failed init");

    {
        let te_s = &mut te_state.borrow_mut();
        let gpu_ = &gpu.as_ref().borrow();
    
        let font = te_s.load_font(String::from("CascadiaCode.ttf"), gpu_).expect("Could not find font");
        te_s.place_text(&font, String::from("Hello"), (10.0, 30.0), TeColor::new(1.0, 1.0, 1.0), 25.0);
        te_s.place_text(&font, String::from("Hello, in red"), (10.0, 60.0), TeColor::new(1.0, 0.0, 0.0), 25.0);
        te_s.place_text(&font, String::from("HELLO"), (10.0, 90.0), TeColor::new(1.0, 0.0, 0.0), 80.0);
    }

    te_player::event_loop::run(event_loop, window, gpu, te_state, Box::new(|_| {}));
}
