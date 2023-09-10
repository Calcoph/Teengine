pub use gilrs;
use gilrs::{Event, Gilrs};
use std::thread;
use winit::event_loop::EventLoopProxy;

pub fn listen(event_loop: EventLoopProxy<Event>) {
    thread::spawn(move || {
        let mut listener = Gilrs::new().expect("Couldn't initialize gilrs");
        loop {
            while let Some(ev) = listener.next_event() {
                event_loop.send_event(ev).expect("Event loop no longer exists");
            }
        }
    });
}
