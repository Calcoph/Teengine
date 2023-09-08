pub mod event_loop;

use image::io::Reader as ImageReader;
pub use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use imgui_winit_support::WinitPlatform;
use std::{cell::RefCell, error::Error, rc::Rc};
use te_renderer::{
    initial_config::InitialConfiguration,
    state::{GpuState, TeState},
};
use te_winit::event_loop::EventLoopBuilder;
use wgpu::CommandEncoder;
pub use winit as te_winit;
use winit::{dpi, event_loop::EventLoop, window};

use te_gamepad::gamepad::{self, ControllerEvent};

use crate::InitError;

pub struct PrepareResult {
    pub event_loop: EventLoop<ControllerEvent>,
    pub gpu: Rc<RefCell<GpuState>>,
    pub window: Rc<RefCell<window::Window>>,
    pub te_state: Rc<RefCell<TeState>>,
    pub context: Context,
    pub platform: WinitPlatform,
    pub renderer: Renderer,
}

/// Get all the structs needed to start the engine, skipping the boilerplate.
pub async fn prepare(
    config: InitialConfiguration,
    log: bool,
) -> Result<PrepareResult, Box<dyn Error>> {
    if log {
        env_logger::init();
    }

    let img = match ImageReader::open(&config.icon_path)?.decode() {
        Ok(img) => Ok(img.to_rgba8()),
        Err(_) => Err(InitError::Unkown),
    }?;

    let event_loop = EventLoopBuilder::with_user_event().build();
    gamepad::listen(event_loop.create_proxy());

    let wb = window::WindowBuilder::new()
        .with_title(&config.window_name)
        .with_inner_size(dpi::LogicalSize::new(
            config.screen_width,
            config.screen_height,
        ))
        .with_window_icon(Some(
            match window::Icon::from_rgba(img.into_raw(), 64, 64) {
                Ok(icon) => icon,
                Err(_) => panic!("Couldn't get icon raw data"),
            },
        ));

    let window = wb.build(&event_loop).expect("Couldn't create window");

    let gpu = GpuState::new(window.inner_size(), &window).await;
    let state = TeState::new(&window, &gpu, config).await;

    let mut context = Context::create();
    let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
    platform.attach_window(
        context.io_mut(),
        &window,
        imgui_winit_support::HiDpiMode::Default,
    );

    let renderer_config = RendererConfig {
        texture_format: gpu.config.format,
        ..Default::default()
    };

    let renderer = Renderer::new(&mut context, &gpu.device, &gpu.queue, renderer_config);

    Ok(PrepareResult {
        event_loop,
        gpu: Rc::new(RefCell::new(gpu)),
        window: Rc::new(RefCell::new(window)),
        te_state: Rc::new(RefCell::new(state)),
        context,
        platform,
        renderer,
    })
}

/// After calling prepare() call new_window() for each extra window.
pub async fn new_window(
    config: InitialConfiguration,
    event_loop: &EventLoop<ControllerEvent>,
) -> Result<
    (
        Rc<RefCell<GpuState>>,
        Rc<RefCell<window::Window>>,
        Rc<RefCell<TeState>>,
    ),
    Box<dyn Error>,
> {
    let img = match ImageReader::open(&config.icon_path)?.decode() {
        Ok(img) => img.to_rgba8(),
        Err(_) => panic!("Couldn't find icon"),
    };

    let wb = window::WindowBuilder::new()
        .with_title(&config.window_name)
        .with_inner_size(dpi::LogicalSize::new(
            config.screen_width,
            config.screen_height,
        ))
        .with_window_icon(Some(
            match window::Icon::from_rgba(img.into_raw(), 64, 64) {
                Ok(icon) => icon,
                Err(_) => panic!("Couldn't get raw data"),
            },
        ));

    let window = wb.build(event_loop).expect("Couldn't create window");

    let gpu = GpuState::new(window.inner_size(), &window).await;
    let state = TeState::new(&window, &gpu, config).await;

    Ok((
        Rc::new(RefCell::new(gpu)),
        Rc::new(RefCell::new(window)),
        Rc::new(RefCell::new(state)),
    ))
}

pub trait ImguiState {
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        window: &window::Window,
        platform: &WinitPlatform,
        context: &mut Context,
        gpu: &GpuState,
        renderer: &mut Renderer,
    ) -> CommandEncoder {
        platform
            .prepare_frame(context.io_mut(), window)
            .expect("Failed to prepare frame");
        let ui = context.frame();

        self.create_ui(&ui);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ImGui Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            renderer
                .render(context.render(), &gpu.queue, &gpu.device, &mut render_pass)
                .expect("Rendering failed");
        }
        encoder
    }

    fn create_ui(&mut self, ui: &Ui);
}
