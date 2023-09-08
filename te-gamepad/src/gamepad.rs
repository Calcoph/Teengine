pub use gilrs_core::AxisInfo;
pub use gilrs_core::EvCode;
pub use gilrs_core::EventType;
use gilrs_core::{Event, Gilrs};
use std::{collections::HashMap, thread};
use winit::event_loop::EventLoopProxy;

pub fn listen(event_loop: EventLoopProxy<ControllerEvent>) {
    thread::spawn(move || {
        let mut listener = Gilrs::new().expect("Couldn't initialize gilrs");
        loop {
            while let Some(ev) = listener.next_event() {
                if ev.event == EventType::Connected {
                    let gamepad = listener.gamepad(ev.id).expect("Unreachable");
                    let mut axes = HashMap::new();
                    for axis in gamepad.axes() {
                        if let Some(axis_info) = gamepad.axis_info(axis.to_owned()) {
                            axes.insert(axis.to_owned(), axis_info.to_owned());
                        }
                    }

                    event_loop
                        .send_event(ControllerEvent::Connected {
                            device_id: ev.id,
                            axes,
                        })
                        .expect("Event loop no longer exists");
                } else {
                    event_loop
                        .send_event(ControllerEvent::new(ev))
                        .expect("Event loop no longer exists");
                }
            }
        }
    });
}

#[derive(Debug)]
pub enum ControllerEvent {
    Connected {
        device_id: usize,
        axes: HashMap<EvCode, AxisInfo>,
    },
    Other {
        device_id: usize,
        event: EventType,
    },
}

impl ControllerEvent {
    fn new(ev: Event) -> ControllerEvent {
        ControllerEvent::Other {
            device_id: ev.id,
            event: ev.event,
        }
    }
}
