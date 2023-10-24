use te_player::{
    event_loop::Event,
    te_winit::{event::WindowEvent, event_loop::ControlFlow},
};
use te_renderer::initial_config::InitialConfiguration;

fn main() {
    pollster::block_on(as_main());
}

async fn as_main() {
    let (event_loop, gpu1, window1, te_state1) = te_player::prepare(
        InitialConfiguration {
            screen_width: 1000,
            screen_height: 500,
            ..InitialConfiguration::default()
        },
        true,
    )
    .await
    .expect("Failed window 1");

    let (gpu2, window2, te_state2) = te_player::new_window(
        InitialConfiguration {
            screen_width: 200,
            screen_height: 500,
            ..InitialConfiguration::default()
        },
        &event_loop,
    )
    .await
    .expect("Failed window 2");

    let (gpu3, window3, te_state3) = te_player::new_window(
        InitialConfiguration {
            screen_width: 300,
            screen_height: 300,
            ..InitialConfiguration::default()
        },
        &event_loop,
    )
    .await
    .expect("Failed window 3");

    let mut last_render_time1 = std::time::Instant::now();
    let mut last_render_time2 = std::time::Instant::now();
    let mut last_render_time3 = std::time::Instant::now();
    event_loop.run(move |event, window_target| {
        window_target.set_control_flow(ControlFlow::Poll);
        match &event {
            Event::WindowEvent { window_id, event } => {
                let res = if *window_id == window1.borrow().id() {
                    Some((&te_state1, &gpu1))
                } else if *window_id == window2.borrow().id() {
                    Some((&te_state2, &gpu2))
                } else if *window_id == window3.borrow().id() {
                    Some((&te_state3, &gpu3))
                } else {
                    None
                };

                if let Some((state, gpu)) = res {
                    match event {
                        WindowEvent::Resized(size) => {
                            gpu.borrow_mut().resize(*size);
                            state.borrow_mut().resize(*size);
                        }
                        WindowEvent::CloseRequested => window_target.exit(),
                        WindowEvent::RedrawRequested => {
                            let res = if *window_id == window1.borrow().id() {
                                Some((&te_state1, &gpu1, &mut last_render_time1))
                            } else if *window_id == window2.borrow().id() {
                                Some((&te_state2, &gpu2, &mut last_render_time2))
                            } else if *window_id == window3.borrow().id() {
                                Some((&te_state3, &gpu3, &mut last_render_time3))
                            } else {
                                None
                            };

                            if let Some((state, gpu, last_render_time)) = res {
                                let now = std::time::Instant::now();
                                let dt = now - *last_render_time;
                                *last_render_time = now;
                                state.borrow_mut().update(dt, &gpu.borrow());
                                let output = gpu
                                    .borrow()
                                    .surface
                                    .get_current_texture()
                                    .expect("Couldn't get surface texture");
                                let view = output
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());
                                let mut encoder = te_renderer::state::TeState::prepare_render(&gpu.borrow());
                                state
                                    .borrow_mut()
                                    .render(&view, &gpu.borrow(), &mut encoder, &vec![]);
                                state.borrow_mut().end_render(&gpu.borrow(), encoder);
                                output.present();
                                state.borrow_mut().text.after_present()
                            }
                        }
                        _ => (),
                    }
                }
            }
            Event::Suspended => window_target.set_control_flow(ControlFlow::Wait), // TODO: confirm that it pauses the game
            Event::Resumed => (), // TODO: confirm that it unpauses the game
            Event::AboutToWait => {
                window1.borrow().request_redraw();
                window2.borrow().request_redraw();
                window3.borrow().request_redraw();
            }
            _ => (),
        }
    }).unwrap()
}
