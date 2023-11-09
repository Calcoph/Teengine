use cgmath::{vec2, point3};
use te_player::{event_loop::Event, te_winit::{event::WindowEvent, event_loop::ControlFlow}};
use te_renderer::{initial_config::InitialConfiguration, state::TeColor};

fn main() {
    pollster::block_on(as_main());
}

async fn as_main() {
    let (event_loop, gpu, window, te_state) = te_player::prepare(
        InitialConfiguration {
            screen_size: vec2(1000, 500),
            camera_pitch: -90.0,
            camera_position: point3(0.0, 10.0, 0.0),
            camera_sensitivity: 10.0,
            camera_speed: 2.0,
            ..InitialConfiguration::default()
        },
        true,
    )
    .await
    .expect("Failed init");

    te_state.borrow_mut().bgcolor = TeColor::new(1.0, 0.0, 0.0);
    te_state.borrow_mut().add_model("box03.glb", &gpu.borrow(), point3(0.0, 0.0, 0.0)).build().unwrap();

    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, window_target| {
        if cfg!(feature = "draw_when_told") {
            window_target.set_control_flow(ControlFlow::Wait)
        } else {
            window_target.set_control_flow(ControlFlow::Poll)
        }
        match &event {
            Event::WindowEvent { window_id, event } if *window_id == window.borrow().id() => {
                match event {
                    WindowEvent::Resized(size) => {
                        gpu.borrow_mut().resize(*size);
                        te_state.borrow_mut().resize(*size);
                    }
                    WindowEvent::CloseRequested => window_target.exit(),
                    WindowEvent::RedrawRequested => {
                        if *window_id == window.borrow().id() {
                            let now = std::time::Instant::now();
                            let dt = now - last_render_time;
                            last_render_time = now;
                            te_state.borrow_mut().update(dt, &gpu.borrow());
                            if let Ok(output) = gpu.borrow().surface.get_current_texture() {
                                let view = output
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());
                                let mut encoder =
                                    te_renderer::state::TeState::prepare_render(&gpu.borrow());
                                te_state
                                    .borrow_mut()
                                    .render_wireframe(&view, &gpu.borrow(), &mut encoder);
                                //te_state.borrow_mut().render(&view, &gpu.borrow(), &mut encoder, &[]);
                                te_state.borrow_mut().end_render(&gpu.borrow(), encoder);
                                output.present();
                                te_state.borrow_mut().text.after_present()
                            }
                        }
                    }
                    _ => {te_state.borrow_mut().input(event);},
                }
            }
            Event::Suspended => window_target.set_control_flow(ControlFlow::Wait),
            Event::Resumed => (),
            Event::AboutToWait => window.borrow().request_redraw(),
            _ => (),
        }

    }).unwrap()
}

