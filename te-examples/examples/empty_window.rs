use cgmath::vec2;
use te_player::event_loop::PlaceholderTextSender;
use te_renderer::initial_config::InitialConfiguration;

fn main() {
    pollster::block_on(as_main());
}

async fn as_main() {
    let (event_loop, gpu, window, te_state) = te_player::prepare(
        InitialConfiguration {
            screen_size: vec2(1000, 500),
            ..InitialConfiguration::default()
        },
        true,
    )
    .await
    .expect("Failed init");

    te_player::event_loop::run(
        event_loop,
        window,
        gpu,
        te_state,
        PlaceholderTextSender::new(),
        Box::new(|_| {}),
    );
}
