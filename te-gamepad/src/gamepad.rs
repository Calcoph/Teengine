use gilrs_core::{ Gilrs, Event, EventType};
use winit::{event_loop::EventLoopProxy};
use std::thread;

pub fn listen(event_loop: EventLoopProxy<ControllerEvent>) {
    thread::spawn(move || {
        let mut listener = Gilrs::new().unwrap();
        loop {
            while let Some(ev) = listener.next_event() {
                event_loop.send_event(ControllerEvent::new(ev)).unwrap();
            }
        }
    });
}

#[derive(Debug, Clone, Copy)]
pub struct ControllerEvent {
    pub device_id: usize,
    pub event: EventType,
}

impl ControllerEvent {
    fn new(ev: Event) -> ControllerEvent {
        ControllerEvent { device_id: ev.id, event: ev.event }
    }
}
