use std::{cell::RefCell, rc::Rc};

use te_gamepad::gamepad::ControllerEvent;
use te_renderer::{state::{TeState, GpuState, Section}, text::FontReference};
use winit::{window::Window, event_loop::{EventLoop, ControlFlow}, event::WindowEvent};
pub use winit::event::Event as Event;

pub fn run<T: TextSender + 'static>(
    event_loop: EventLoop<ControllerEvent>,
    window: Rc<RefCell<Window>>,
    gpu: Rc<RefCell<GpuState>>,
    state: Rc<RefCell<TeState>>,
    text: Rc<RefCell<T>>,
    mut event_handler: Box<dyn FnMut(Event<ControllerEvent>)>,
) {
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _window_target, control_flow| {
        *control_flow = ControlFlow::Poll;
        match &event {
            Event::WindowEvent { window_id, event } if *window_id == window.borrow().id() => {
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
                    _ => (),
                }
            },
            Event::Suspended => *control_flow = ControlFlow::Wait, // TODO: confirm that it pauses the game
            Event::Resumed => (), // TODO: confirm that it unpauses the game
            Event::MainEventsCleared => window.borrow().request_redraw(),
            Event::RedrawRequested(window_id) => if *window_id == window.borrow().id() {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.borrow_mut().update(dt, &gpu.borrow());
                let output = gpu.borrow().surface.get_current_texture().unwrap();
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = te_renderer::state::TeState::prepare_render(&gpu.borrow());
                state.borrow_mut().render(&view, &gpu.borrow(), &mut encoder, text.borrow_mut().get_text());
                state.borrow_mut().end_render(&gpu.borrow(), encoder);
                output.present();
                state.borrow_mut().text.after_present()
            },
            _ => ()
        }

        event_handler(event);
    })
}

pub trait TextSender {
    fn get_text<'a>(&'a mut self) -> &'a [(FontReference, Vec<Section<'a>>)];
}

pub struct PlaceholderTextSender {
    v: Vec<(te_renderer::text::FontReference, Vec<te_renderer::state::Section<'static>>)>
}

impl PlaceholderTextSender {
    pub fn new() -> Rc<RefCell<PlaceholderTextSender>> {
        Rc::new(RefCell::new(PlaceholderTextSender { v: vec![] }))
    }
}

impl TextSender for PlaceholderTextSender {
    fn get_text<'a>(&'a mut self) -> &'a [(te_renderer::text::FontReference, Vec<te_renderer::state::Section<'a>>)] {
        &self.v
    }
}

