use std::io::Error;

use image::io::Reader as ImageReader;
use winit::{event_loop::{EventLoop, ControlFlow}, window::{WindowBuilder, Icon}, dpi, event::{Event, WindowEvent}};

use te_gamepad::gamepad;
use main::consts as c;
use main::mapmaker;

pub async fn run() -> Result<(), Error> {
    env_logger::init();
    let img = match ImageReader::open("icon.png")?.decode() {
        Ok(img) => img.to_rgba8(),
        Err(_) => panic!("Couldn't find icon"),
    };

    let event_loop = EventLoop::with_user_event();
    gamepad::listen(event_loop.create_proxy());
    let wb = WindowBuilder::new()
        .with_title("Tilengine")
        .with_inner_size(dpi::LogicalSize::new(c::SCREEN_W, c::SCREEN_H))
        .with_window_icon(Some(match Icon::from_rgba(img.into_raw(), 64, 64) {
            Ok(icon) => icon,
            Err(_) => panic!("Couldn't get raw data")
        }));

    let window = wb.build(&event_loop)
        .unwrap();

    let mut mapmaker = mapmaker::ImguiState::new(&window).await;
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
                    WindowEvent::ReceivedCharacter(character) => (),//println!("TODO: ReceivedCharacter{:?}", character),
                    WindowEvent::Focused(is_focused) => (),//println!("TODO: Focused {:?}", is_focused), // TODO
                    WindowEvent::KeyboardInput { device_id: _, input: _, is_synthetic: _ } => {mapmaker.input(&event);}, // TODO
                    WindowEvent::ModifiersChanged(modifiers) => (),//println!("TODO: ModifiersChanged ({:?})", modifiers), // TODO
                    WindowEvent::CursorMoved { device_id: _, position: _, .. } => {mapmaker.input(&event);},
                    WindowEvent::CursorEntered { device_id } => (),//println!("TODO: CursorEntered ({:?})", device_id), // TODO
                    WindowEvent::CursorLeft { device_id } => (),//println!("TODO: CursorLeft ({:?})", device_id), // TODO
                    WindowEvent::MouseWheel { .. } => {mapmaker.input(&event);},
                    WindowEvent::MouseInput { .. } => {mapmaker.input(&event);},
                    WindowEvent::TouchpadPressure { device_id, pressure, stage } => (),//println!("Ignoring TouchpadPressure ({:?}, {:?}, {:?})", device_id, pressure, stage), // ignore until I know if it's useful
                    WindowEvent::AxisMotion { device_id, axis, value } => (),//println!("Ignoring AxisMotion ({:?}, {:?}, {:?})", device_id, axis, value), // ignore until I know if it's useful
                    WindowEvent::Touch(touch) => (),//println!("TODO: Touch ({:?})", touch), // TODO: do the same as mouse click event
                    WindowEvent::ScaleFactorChanged { scale_factor: _, new_inner_size } => mapmaker.resize(**new_inner_size),
                    WindowEvent::ThemeChanged(theme) => (),//println!("TODO: ThemeChanged ({:?})", theme), // TODO
                }
            },
            Event::DeviceEvent { device_id, event } => {
                match event {
                    winit::event::DeviceEvent::Added => (),//println!("TODO: Device Added ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::Removed => (),//println!("TODO: Device Removed ({:?})", device_id), // TODO
                    winit::event::DeviceEvent::MouseMotion { delta } => (),//println!("TODO: Mouse Moved ({:?}, {:?})", device_id, delta),
                    winit::event::DeviceEvent::MouseWheel { delta } => (),//println!("TODO: Mouse Wheel ({:?}, {:?})", device_id, delta), // TODO
                    winit::event::DeviceEvent::Motion { axis, value } => (),//println!("TODO: Device Motion ({:?}, {:?}, {:?})", device_id, value, axis), // TODO
                    winit::event::DeviceEvent::Button { button, state } => (),//println!("TODO: Device Button ({:?}, {:?}, {:?})", device_id, button, state), // TODO
                    winit::event::DeviceEvent::Key(input) => (),//println!("TODO: Device Key ({:?}, {:?})", device_id, input), // TODO
                    winit::event::DeviceEvent::Text { codepoint } => (),//println!("TODO: Device Text ({:?}, {:?})", device_id, codepoint), // TODO
                }
            },
            Event::UserEvent(event) => {
                match event.event {
                    gilrs_core::EventType::ButtonPressed(code) => (),//println!("TODO: ButtonPressed ({:?}, {:?})", event.device_id, code), // TODO
                    gilrs_core::EventType::ButtonReleased(code) => (),//println!("TODO: ButtonReleased ({:?}, {:?})", event.device_id, code), // TODO
                    gilrs_core::EventType::AxisValueChanged(value, code) => (),//println!("TODO: AxisValueChanged ({:?}, {:?}, {:?})", event.device_id, code, value), // TODO
                    gilrs_core::EventType::Connected => (),//println!("TODO: Connected ({:?})", event.device_id), // TODO
                    gilrs_core::EventType::Disconnected => (),//println!("TODO: Disconnected ({:?})", event.device_id), // TODO
                }
            },
            Event::Suspended => *control_flow = ControlFlow::Wait, // TODO: confirm that it pauses the game
            Event::Resumed => (), // TODO: confirm that it unpauses the game
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(window_id) => if *window_id == window.id() {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                mapmaker.update(dt);
                match mapmaker.render(&window) {
                    Ok(_) => {},
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => mapmaker.resize(mapmaker.state.size),
                    // The system is out of memory, we should quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e), 
                }
            },
            Event::RedrawEventsCleared => (), // TODO
            Event::LoopDestroyed => (),//println!("TODO: LoopDestroyed"), // TODO: clean memory or save settings or whatever
            _ => () // ignore windowevents that aren't for current window
        }

        mapmaker.platform.handle_event(mapmaker.context.io_mut(), &window, &event)
    })
}
