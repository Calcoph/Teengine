use std::{cell::RefCell, rc::Rc};

use te_gamepad::gamepad::ControllerEvent;
use te_renderer::state::{State, GpuState};
use winit::{window::Window, event_loop::{EventLoop, ControlFlow}, event::WindowEvent};
pub use winit::event::Event as Event;

pub fn run(event_loop: EventLoop<ControllerEvent>, window: Window, gpu: Rc<RefCell<GpuState>>, state: Rc<RefCell<State>>, mut event_handler: Box<dyn FnMut(Event<ControllerEvent>)>) {
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _window_target, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent { window_id, event } if *window_id == window.id() => {
                match event {
                    WindowEvent::Resized(size) => {
                        gpu.borrow_mut().resize(*size);
                        state.borrow_mut().resize(*size);
                    },
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit, //control_flow is a pointer to the next action we wanna do. In this case, exit the program
                    WindowEvent::ScaleFactorChanged { scale_factor: _, new_inner_size } => {
                        gpu.borrow_mut().resize(**new_inner_size);
                        state.borrow_mut().resize(**new_inner_size)
                    },
                    _ => ()
                }
            },
            Event::Suspended => *control_flow = ControlFlow::Wait, // TODO: confirm that it pauses the game
            Event::Resumed => (), // TODO: confirm that it unpauses the game
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(window_id) => if *window_id == window.id() {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.borrow_mut().update(dt, &gpu.borrow());
                let output = gpu.borrow().surface.get_current_texture().unwrap();
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                state.borrow_mut().render(&view, &gpu.borrow()).unwrap();
                output.present();
            },
            _ => ()
        }

        event_handler(event)
    })
}