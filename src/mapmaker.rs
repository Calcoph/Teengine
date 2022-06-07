use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::{window::Window, event::WindowEvent};
use wgpu;

use crate::{state::{State, GpuState}, camera::Projection};

pub struct ImguiState {
    gpu: GpuState,
    pub context: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    pub state: State,
    files: Vec<String>,
    camera_controls_win: CameraControlsWin,
    model_selector_win: ModelSelectorWin
}

struct CameraControlsWin {
    show_hotkeys: bool,
    fovy: f32
}

struct ModelSelectorWin {
    search_str: String
}

impl ImguiState {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let gpu = GpuState::new(size, window).await;
        let state = State::new(window, &gpu).await;
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);

        let renderer_config = RendererConfig {
            texture_format: gpu.config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &gpu.device, &gpu.queue, renderer_config);

        let files = get_file_names();

        let camera_controls_win = CameraControlsWin {
            show_hotkeys: false,
            fovy: 20.0
        };

        let model_selector_win = ModelSelectorWin {
            search_str: String::new()
        };

        ImguiState {
            gpu,
            context,
            platform,
            renderer,
            state,
            files,
            camera_controls_win,
            model_selector_win
        }
    }

    pub fn render_imgui(&mut self, view: &wgpu::TextureView, window: &Window) {
        self.platform.prepare_frame(self.context.io_mut(), window).expect("Failed to prepare frame");
        let ui = self.context.frame();
        {
            let mut opened = false;
            imgui::Window::new("Camera controls")
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(&ui, || {
                    let camera_cont = &mut self.state.camera.camera_controller;
                    let projection = &mut self.state.camera.projection;
                    let state = &mut self.camera_controls_win;
                    Slider::new("speed", 0.0 as f32, 1000.0 as f32).build(&ui, &mut camera_cont.speed);
                    Slider::new("sensitivity", 0.0 as f32, 20.0 as f32).build(&ui, &mut camera_cont.sensitivity);
                    Slider::new("fovy", 1.0 as f32, 45.0 as f32).build(&ui, &mut state.fovy);
                    *projection = projection.change_fovy(cgmath::Deg(state.fovy));
                    Slider::new("yaw", -4.0 as f32, 4.0 as f32).build(&ui, &mut camera_cont.rotate_horizontal);
                    VerticalSlider::new("pitch", [20.0, 100.0], -4.0 as f32, 4.0 as f32)
                        .build(&ui, &mut camera_cont.rotate_vertical);
                    ui.same_line();
                    VerticalSlider::new("zooom", [20.0, 100.0], -4.0 as f32, 4.0 as f32)
                        .build(&ui, &mut camera_cont.scroll);

                    ui.checkbox("Show camera hotkeys", &mut state.show_hotkeys);
                    if state.show_hotkeys {
                        ui.text("press WASD to move");
                        ui.text("press space/shift to move up/down");
                        ui.text("press QE to rotate yaw");
                        ui.text("press ZX to rotate pitch");
                        ui.text("press RF to zoom in/out");
                    }
                });
            imgui::Window::new("Models")
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(&ui, || {
                    let state = &mut self.model_selector_win;
                    ui.text("Put your gltf/glb models in\nignore/resources/ so they show up here");
                    if ui.button("Refresh") {
                        self.files = get_file_names();
                    };
                    ui.separator();
                    ui.input_text("search", &mut state.search_str).build();
                    for file_name in &self.files {
                        let name = match file_name.strip_suffix(".glb") {
                            Some(n) => n,
                            None => file_name.strip_suffix(".gltf").unwrap(),
                        };
                        if name.contains(&state.search_str) {
                            if ui.button(name) {
                                self.state.change_model(file_name, &self.gpu)
                            };
                        }
                    };
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
        if new_size.width > 0 && new_size.height > 0 {
            self.gpu.resize(new_size);
            self.state.resize(new_size)
        }
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
        self.state.render(&view, &self.gpu)?;
        self.render_imgui(&view, window);
        output.present();

        Ok(())
    }
}

fn get_file_names() -> Vec<String> {
    let mut names = Vec::new();
    match std::fs::read_dir(std::path::Path::new("ignore/resources")) {
        Ok(files) => for file in files {
            let file_name = file.unwrap().file_name().to_str().unwrap().to_string();
            if file_name.ends_with(".glb") || file_name.ends_with(".gltf") {
                names.push(file_name)
            }
        },
        Err(_) => names.push(String::from("Error accessing the directory ignore/resources/")),
    };

    names
}