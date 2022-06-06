use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::{window::Window, event::WindowEvent};
use wgpu;

use crate::state::{State, GpuState};

pub struct ImguiState {
    gpu: GpuState,
    pub context: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    pub state: State
}

impl ImguiState {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let gpu = GpuState::new(size, window).await;
        let state = State::new(window, &gpu).await;
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
        context.set_ini_filename(None);

        let renderer_config = RendererConfig {
            texture_format: gpu.config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &gpu.device, &gpu.queue, renderer_config);


        ImguiState { gpu, context, platform, renderer, state }
    }

    pub fn render_imgui(&mut self, view: &wgpu::TextureView, window: &Window) {
        self.platform.prepare_frame(self.context.io_mut(), window).expect("Failed to prepare frame");
        let ui = self.context.frame();
        {
            let mut opened = false;
            let window = imgui::Window::new("Hello too");
            window
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(&ui, || {
                    ui.text(format!("Frametime: {:?}", "aaa"));
                });
            ui.show_demo_window(&mut opened);
        }

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ImGui Render Encoder")
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            self.renderer.render(ui.render(), &self.gpu.queue, &self.gpu.device, &mut render_pass).expect("Rendering failed");
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.state.resize(new_size, &mut self.gpu)
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.state.input(event)
    }

    pub fn update(&mut self, dt: std::time::Duration) {
        self.state.update(dt, &self.gpu);
    }

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.gpu.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.state.render(window, &view, &self.gpu)?;
        self.render_imgui(&view, window);
        output.present();

        Ok(())
    }
}