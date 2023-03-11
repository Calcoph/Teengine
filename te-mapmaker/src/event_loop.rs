use std::{error::Error, fmt::Display};

use image::io::Reader as ImageReader;
use winit::{event_loop::{EventLoop, ControlFlow}, window::{WindowBuilder, Icon}, dpi, event::{Event, WindowEvent}};

use te_renderer::initial_config::InitialConfiguration;
use te_gamepad::gamepad;

use crate::mapmaker;

#[derive(Debug)]
enum InitError {
    Unkown,
    Opaque
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

    let event_loop = EventLoop::with_user_event();
    gamepad::listen(event_loop.create_proxy());
    let wb = WindowBuilder::new()
        .with_title("Tilengine")
        .with_inner_size(dpi::LogicalSize::new(config.screen_width, config.screen_height))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => Ok(icon),
            Err(_) => Err(InitError::Opaque)
        }?));

    let window = wb.build(&event_loop)
        .unwrap();

    let mut mapmaker = mapmaker::ImguiState::new(&window, config.clone(), default_model).await;
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _window_target, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::NewEvents(_cause) => (), // TODO
            Event::WindowEvent { window_id, event } if *window_id == window.id() => {
                match event {
                    WindowEvent::Resized(size) => {
                        mapmaker.resize(size.clone());
                    },
                    WindowEvent::Moved(_) => (), // ignore
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit, //control_flow is a poionter to the next action we wanna do. In this case, exit the program
                    WindowEvent::Destroyed => (),//println!("TODO: Destroyed window {:?}", window_id), // TODO
                    WindowEvent::DroppedFile(_) => (), // ignore
                    WindowEvent::HoveredFile(_) => (), // ignore
                    WindowEvent::HoveredFileCancelled => (), // ignore
                    WindowEvent::ReceivedCharacter(_character) => (),//println!("TODO: ReceivedCharacter{:?}", character),
                    WindowEvent::Focused(_is_focused) => (),//println!("TODO: Focused {:?}", is_focused), // TODO
                    WindowEvent::KeyboardInput { device_id: _, input: _, is_synthetic: _ } => {mapmaker.input(&event);}, // TODO
                    WindowEvent::ModifiersChanged(_modifiers) => (),//println!("TODO: ModifiersChanged ({:?})", modifiers), // TODO
                    WindowEvent::CursorMoved { device_id: _, position: _, .. } => {mapmaker.input(&event);},
                    WindowEvent::CursorEntered { device_id: _device_id } => (),//println!("TODO: CursorEntered ({:?})", device_id), // TODO
                    WindowEvent::CursorLeft { device_id: _device_id } => (),//println!("TODO: CursorLeft ({:?})", device_id), // TODO
                    WindowEvent::MouseWheel { .. } => {mapmaker.input(&event);},
                    WindowEvent::MouseInput { .. } => {mapmaker.input(&event);},
                    WindowEvent::TouchpadPressure { device_id: _device_id, pressure: _pressure, stage: _stage } => (),//println!("Ignoring TouchpadPressure ({:?}, {:?}, {:?})", device_id, pressure, stage), // ignore until I know if it's useful
                    WindowEvent::AxisMotion { device_id: _device_id, axis: _axis, value: _value } => (),//println!("Ignoring AxisMotion ({:?}, {:?}, {:?})", device_id, axis, value), // ignore until I know if it's useful
                    WindowEvent::Touch(_touch) => (),//println!("TODO: Touch ({:?})", touch), // TODO: do the same as mouse click event
                    WindowEvent::ScaleFactorChanged { scale_factor: _, new_inner_size } => mapmaker.resize(**new_inner_size),
                    WindowEvent::ThemeChanged(_theme) => (),//println!("TODO: ThemeChanged ({:?})", theme), // TODO
                    WindowEvent::Ime(_) => (), // TODO
                    WindowEvent::Occluded(_) => (), // TODO
                }
            },
            Event::DeviceEvent { device_id: _device_id, event } => {
                match event {
                    winit::event::DeviceEvent::Added => (),//println!("TODO: Device Added ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::Removed => (),//println!("TODO: Device Removed ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::MouseMotion { delta: _delta } => (),//println!("TODO: Mouse Moved ({:?}, {:?})", device_id, delta),
                    winit::event::DeviceEvent::MouseWheel { delta: _delta } => (),//println!("TODO: Mouse Wheel ({:?}, {:?})", device_id, delta), // TODO
                    winit::event::DeviceEvent::Motion { axis: _axis, value: _value } => (),//println!("TODO: Device Motion ({:?}, {:?}, {:?})", device_id, value, axis), // TODO
                    winit::event::DeviceEvent::Button { button: _button, state: _state } => (),//println!("TODO: Device Button ({:?}, {:?}, {:?})", device_id, button, state), // TODO
                    winit::event::DeviceEvent::Key(_input) => (),//println!("TODO: Device Key ({:?}, {:?})", device_id, input), // TODO
                    winit::event::DeviceEvent::Text { codepoint: _codepoint } => (),//println!("TODO: Device Text ({:?}, {:?})", device_id, codepoint), // TODO
                }
            },
            Event::UserEvent(event) => {
                match event.event {
                    gilrs_core::EventType::ButtonPressed(_code) => (),//println!("TODO: ButtonPressed ({:?}, {:?})", event.device_id, code), // TODO
                    gilrs_core::EventType::ButtonReleased(_code) => (),//println!("TODO: ButtonReleased ({:?}, {:?})", event.device_id, code), // TODO
                    gilrs_core::EventType::AxisValueChanged(_value, _code) => (),//println!("TODO: AxisValueChanged ({:?}, {:?}, {:?})", event.device_id, code, value), // TODO
                    gilrs_core::EventType::Connected => (),//println!("TODO: Connected ({:?})", event.device_id), // TODO
                    gilrs_core::EventType::Disconnected => (),//println!("TODO: Disconnected ({:?})", event.device_id), // TODO
                }
            },
            Event::Suspended => *control_flow = ControlFlow::Wait,
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(window_id) => if *window_id == window.id() {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                mapmaker.update(dt);
                match mapmaker.render(&window, config.tile_size, &config.resource_files_directory, &config.map_files_directory) {
                    Ok(_) => {},
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => mapmaker.resize(mapmaker.state.size),
                    // The system is out of memory, we should quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e), 
                }
            },
            _ => () // ignore windowevents that aren't for current window
        }

        mapmaker.platform.handle_event(mapmaker.context.io_mut(), &window, &event)
    })
}
