use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::window::Window;
use wgpu;

pub struct ImguiState {
    pub context: Context,
    pub platform: WinitPlatform,
    renderer: Renderer
}

impl ImguiState {
    pub fn new(window: &Window, config: &wgpu::SurfaceConfiguration, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
        context.set_ini_filename(None);

        let renderer_config = RendererConfig {
            texture_format: config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, device, queue, renderer_config);

        ImguiState { context, platform, renderer}
    }

    pub fn render(&mut self, queue: &wgpu::Queue, device: &wgpu::Device, view: &wgpu::TextureView, window: &Window) {
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

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
            self.renderer.render(ui.render(), &queue, &device, &mut render_pass).expect("Rendering failed");
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}