use gilrs_core::{Gilrs, Event};
pub use gilrs_core::EvCode as EvCode;
pub use gilrs_core::AxisInfo as AxisInfo;
pub use gilrs_core::EventType as EventType;
use winit::{event_loop::EventLoopProxy};
use std::{thread, collections::HashMap};

pub fn listen(event_loop: EventLoopProxy<ControllerEvent>) {
    thread::spawn(move || {
        let mut listener = Gilrs::new().unwrap();
        loop {
            while let Some(ev) = listener.next_event() {
                if ev.event == EventType::Connected {
                    let gamepad = listener.gamepad(ev.id).unwrap();
                    let mut axes = HashMap::new();
                    for axis in gamepad.axes() {
                        if let Some(axis_info) = gamepad.axis_info(axis.to_owned()) {
                            axes.insert(axis.to_owned(), axis_info.to_owned());
                        }
                    }

                    event_loop.send_event(ControllerEvent::Connected {
                        device_id: ev.id,
                        axes
                    }).unwrap();

                } else {
                    event_loop.send_event(ControllerEvent::new(ev)).unwrap();
                }
            }
        }
    });
}

#[derive(Debug)]
pub enum ControllerEvent {
    Connected {
        device_id: usize,
        axes: HashMap<EvCode, AxisInfo>
    },
    Other {
        device_id: usize,
        event: EventType
    }
}

impl ControllerEvent {
    fn new(ev: Event) -> ControllerEvent {
        ControllerEvent::Other { device_id: ev.id, event: ev.event }
    }
}
