use std::{cell::RefCell, rc::Rc};

use imgui::{Context, Io};
use imgui_wgpu::Renderer;
use imgui_winit_support::WinitPlatform;
use te_gamepad::gamepad::gilrs::Event as GEvent;
use te_renderer::state::{GpuState, TeState};
pub use winit::event::Event;
use winit::{
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::event_loop::TextSender;

use super::ImguiState;

#[cfg(feature = "draw_when_told")]
type EventHandler<I> = Box<dyn FnMut(Event<GEvent>, &mut I, &mut Io) -> bool>;
#[cfg(not(feature = "draw_when_told"))]
type EventHandler<I> = Box<dyn FnMut(Event<GEvent>, &mut I, &mut Io)>;

pub fn run<I: ImguiState + 'static, T: TextSender + 'static>(
    event_loop: EventLoop<GEvent>,
    window: Rc<RefCell<Window>>,
    gpu: Rc<RefCell<GpuState>>,
    state: Rc<RefCell<TeState>>,
    mut imgui_state: I,
    mut platform: WinitPlatform,
    mut context: Context,
    mut renderer: Renderer,
    text: Rc<RefCell<T>>,
    mut event_handler: EventHandler<I>,
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
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit, //control_flow is a pointer to the next action we wanna do. In this case, exit the program
                    WindowEvent::ScaleFactorChanged {
                        scale_factor: _,
                        new_inner_size,
                    } => {
                        gpu.borrow_mut().resize(**new_inner_size);
                        state.borrow_mut().resize(**new_inner_size)
                    }
                    _ => (),
                }
            }
            Event::Suspended => *control_flow = ControlFlow::Wait,
            Event::MainEventsCleared =>
            {
                #[cfg(not(feature = "draw_when_told"))]
                window.borrow().request_redraw()
            }
            Event::RedrawRequested(window_id) => {
                if *window_id == window.borrow().id() {
                    let now = std::time::Instant::now();
                    let dt = now - last_render_time;
                    last_render_time = now;

                    state.borrow_mut().update(dt, &gpu.borrow());
                    let output = gpu
                        .borrow()
                        .surface
                        .get_current_texture()
                        .expect("Couldn't get surface texture");
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoders = te_renderer::state::TeState::prepare_render(&gpu.borrow());

                    let imgui_encoder = imgui_state.render(
                        &view,
                        &window.borrow(),
                        &platform,
                        &mut context,
                        &gpu.borrow(),
                        &mut renderer,
                    );
                    text.borrow_mut().draw_text(|text| {
                        state
                            .borrow_mut()
                            .render(&view, &gpu.borrow(), &mut encoders, text);
                    });

                    encoders.push(imgui_encoder);
                    state.borrow_mut().end_render(&gpu.borrow(), encoders);
                    output.present();
                    state.borrow_mut().text.after_present();
                }
            }
            _ => (),
        }

        platform.handle_controller_event(context.io_mut(), &window.borrow(), &event);
        #[cfg(feature = "draw_when_told")]
        if event_handler(event, &mut imgui_state, context.io_mut()) {
            window.borrow().request_redraw()
        }
        #[cfg(not(feature = "draw_when_told"))]
        event_handler(event, &mut imgui_state, context.io_mut());
    })
}
