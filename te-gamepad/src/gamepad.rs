pub use gilrs;
use gilrs::{Event, Gilrs};
#[cfg(feature = "imgui")]
use imgui_gilrs::GamepadHandler;
use std::thread;
use winit::event_loop::EventLoopProxy;

#[cfg(feature = "imgui")]
pub fn listen(event_loop: EventLoopProxy<Event>) -> GamepadHandler {
    listener(event_loop);

    GamepadHandler::new()
}

#[cfg(not(feature = "imgui"))]
pub fn listen(event_loop: EventLoopProxy<Event>) {
    listener(event_loop);
}

fn listener(event_loop: EventLoopProxy<Event>) {
    thread::spawn(move || {
        let mut listener = Gilrs::new().expect("Couldn't initialize gilrs");
        loop {
            while let Some(ev) = listener.next_event() {
                event_loop.send_event(ev).expect("Event loop no longer exists");
            }
        }
    });
}
