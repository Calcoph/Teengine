use std::{error::Error, fmt::Display};

use image::io::Reader as ImageReader;
use winit::{
    dpi,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::{Icon, WindowBuilder},
};

use te_gamepad::gamepad;
use te_renderer::initial_config::InitialConfiguration;

use crate::mapmaker;

#[derive(Debug)]
enum InitError {
    Unkown,
    Opaque,
}

impl Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            InitError::Opaque => "InitError: icon.png must be a transparent png",
            InitError::Unkown => "InitError: Unkown error",
        };

        write!(f, "{}", msg)
    }
}

impl Error for InitError {}

pub async fn run(config: InitialConfiguration, default_model: &str) -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let img = match ImageReader::open("icon.png")?.decode() {
        Ok(img) => Ok(img.to_rgba8()),
        Err(_) => Err(InitError::Unkown),
    }?;

    let event_loop = EventLoopBuilder::with_user_event().build().expect("Couldn't create event loop");
    let mut gamepad_handler = gamepad::listen(event_loop.create_proxy());
    let wb = WindowBuilder::new()
        .with_title("Tilengine")
        .with_inner_size(dpi::LogicalSize::new(
            config.screen_size.x,
            config.screen_size.y,
        ))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => Ok(icon),
            Err(_) => Err(InitError::Opaque),
        }?));

    let window = wb.build(&event_loop).expect("Unable to create window");

    let mut mapmaker = mapmaker::ImguiState::new(&window, config.clone(), default_model).await;
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, window_target| {
        window_target.set_control_flow(ControlFlow::Poll);
        match &event {
            Event::NewEvents(_cause) => (), // TODO
            Event::WindowEvent { window_id, event } if *window_id == window.id() => {
                match event {
                    WindowEvent::Resized(size) => {
                        mapmaker.resize(size.clone());
                    }
                    WindowEvent::Moved(_) => (), // ignore
                    WindowEvent::CloseRequested => window_target.exit(),
                    WindowEvent::Destroyed => (), //println!("TODO: Destroyed window {:?}", window_id), // TODO
                    WindowEvent::DroppedFile(_) => (), // ignore
                    WindowEvent::HoveredFile(_) => (), // ignore
                    WindowEvent::HoveredFileCancelled => (), // ignore
                    WindowEvent::Focused(_is_focused) => (), //println!("TODO: Focused {:?}", is_focused), // TODO
                    WindowEvent::KeyboardInput {
                        ..
                    } => {
                        mapmaker.input(&event);
                    } // TODO
                    WindowEvent::ModifiersChanged(_modifiers) => (), //println!("TODO: ModifiersChanged ({:?})", modifiers), // TODO
                    WindowEvent::CursorMoved {
                        device_id: _,
                        position: _,
                        ..
                    } => {
                        mapmaker.input(&event);
                    }
                    WindowEvent::CursorEntered {
                        device_id: _device_id,
                    } => (), //println!("TODO: CursorEntered ({:?})", device_id), // TODO
                    WindowEvent::CursorLeft {
                        device_id: _device_id,
                    } => (), //println!("TODO: CursorLeft ({:?})", device_id), // TODO
                    WindowEvent::MouseWheel { .. } => {
                        mapmaker.input(&event);
                    }
                    WindowEvent::MouseInput { .. } => {
                        mapmaker.input(&event);
                    }
                    WindowEvent::TouchpadPressure {
                        device_id: _device_id,
                        pressure: _pressure,
                        stage: _stage,
                    } => (), //println!("Ignoring TouchpadPressure ({:?}, {:?}, {:?})", device_id, pressure, stage), // ignore until I know if it's useful
                    WindowEvent::AxisMotion {
                        device_id: _device_id,
                        axis: _axis,
                        value: _value,
                    } => (), //println!("Ignoring AxisMotion ({:?}, {:?}, {:?})", device_id, axis, value), // ignore until I know if it's useful
                    WindowEvent::Touch(_touch) => (), //println!("TODO: Touch ({:?})", touch), // TODO: do the same as mouse click event
                    WindowEvent::ThemeChanged(_theme) => (), //println!("TODO: ThemeChanged ({:?})", theme), // TODO
                    WindowEvent::Ime(_) => (),               // TODO
                    WindowEvent::Occluded(_) => (),
                    WindowEvent::TouchpadMagnify { .. } => (),
                    WindowEvent::SmartMagnify { .. } => (),
                    WindowEvent::TouchpadRotate { .. } => (),          // TODO
                    WindowEvent::RedrawRequested => {
                        if *window_id == window.id() {
                            let now = std::time::Instant::now();
                            let dt = now - last_render_time;
                            last_render_time = now;
                            mapmaker.update(dt);
                            match mapmaker.render(
                                &window,
                                config.tile_size,
                                &config.resource_files_directory,
                                &config.map_files_directory,
                            ) {
                                Ok(_) => {}
                                // Reconfigure the surface if lost
                                Err(wgpu::SurfaceError::Lost) => mapmaker.resize(mapmaker.state.size),
                                // The system is out of memory, we should quit
                                Err(wgpu::SurfaceError::OutOfMemory) => window_target.exit(),
                                // All other errors (Outdated, Timeout) should be resolved by the next frame
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                    }
                    _ => ()
                }
            }
            Event::DeviceEvent {
                device_id: _device_id,
                event,
            } => {
                match event {
                    winit::event::DeviceEvent::Added => (), //println!("TODO: Device Added ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::Removed => (), //println!("TODO: Device Removed ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::MouseMotion { delta: _delta } => (), //println!("TODO: Mouse Moved ({:?}, {:?})", device_id, delta),
                    winit::event::DeviceEvent::MouseWheel { delta: _delta } => (), //println!("TODO: Mouse Wheel ({:?}, {:?})", device_id, delta), // TODO
                    winit::event::DeviceEvent::Motion {
                        axis: _axis,
                        value: _value,
                    } => (), //println!("TODO: Device Motion ({:?}, {:?}, {:?})", device_id, value, axis), // TODO
                    winit::event::DeviceEvent::Button {
                        button: _button,
                        state: _state,
                    } => (), //println!("TODO: Device Button ({:?}, {:?}, {:?})", device_id, button, state), // TODO
                    winit::event::DeviceEvent::Key(_input) => (), //println!("TODO: Device Key ({:?}, {:?})", device_id, input), // TODO
                }
            }
            Event::Suspended => window_target.set_control_flow(ControlFlow::Wait),
            Event::AboutToWait => window.request_redraw(),
            _ => (), // ignore windowevents that aren't for current window
        }

        gamepad_handler.handle_event(mapmaker.context.io_mut(), &window, &mut mapmaker.platform, &event)
    }).unwrap();

    Ok(())
}
