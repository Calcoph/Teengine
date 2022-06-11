use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::{window::Window, event::WindowEvent};
use wgpu;

use te_renderer::state::{State, GpuState};

pub struct ImguiState {
    gpu: GpuState,
    pub context: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    pub state: State,
    resources: Vec<String>,
    maps: Vec<String>,
    camera_controls_win: CameraControlsWin,
    model_selector_win: ModelSelectorWin,
    object_control_win: ObjectControlWin
}

struct CameraControlsWin {
    show_hotkeys: bool,
    fovy: f32
}

struct ModelSelectorWin {
    search_str_gltf: String,
    search_str_temap: String,
    save_map_name: String
}

struct ObjectControlWin {
    _blinking: bool // TODO: Remove unused variable
}

impl ImguiState {
    pub async fn new(
        window: &Window,
        camera_position: (f32, f32, f32),
        camera_yaw: f32,
        camera_pitch: f32,
        camera_fovy: f32,
        camera_znear: f32,
        camera_zfar: f32,
        camera_speed: f32,
        camera_sensitivity: f32,
        resource_files_directory: &str,
        map_files_directory: &str,
        default_texture_path: &str
    ) -> Self {
        let size = window.inner_size();
        let gpu = GpuState::new(size, window).await;
        let state = State::new(
            window,
            &gpu,
            camera_position,
            camera_yaw,
            camera_pitch,
            camera_fovy,
            camera_znear,
            camera_zfar,
            camera_speed,
            camera_sensitivity,
            resource_files_directory.to_string(),
            default_texture_path
        ).await;
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);

        let renderer_config = RendererConfig {
            texture_format: gpu.config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &gpu.device, &gpu.queue, renderer_config);

        let resources = get_resource_names(resource_files_directory);

        let maps = get_map_names(map_files_directory);

        let camera_controls_win = CameraControlsWin {
            show_hotkeys: false,
            fovy: camera_fovy
        };

        let model_selector_win = ModelSelectorWin {
            search_str_gltf: String::new(),
            search_str_temap: String::new(),
            save_map_name: String::new()
        };

        let object_control_win = ObjectControlWin {
            _blinking: true
        };

        ImguiState {
            gpu,
            context,
            platform,
            renderer,
            state,
            resources,
            maps,
            camera_controls_win,
            model_selector_win,
            object_control_win
        }
    }

    pub fn render_imgui(&mut self, view: &wgpu::TextureView, window: &Window, tile_size: (f32, f32, f32), resource_files_directory: &str, map_files_directory: &str, default_texture_path: &str) {
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
                    ui.separator();
                    ChildWindow::new("models")
                        .size([250.0, 0.0])
                        .build(&ui, || {
                        if ui.button("Refresh") {
                            self.resources = get_resource_names(resource_files_directory);
                        };
                        ui.input_text("search", &mut state.search_str_gltf).build();
                        for file_name in &self.resources {
                            let name = match file_name.strip_suffix(".glb") {
                                Some(n) => n,
                                None => match file_name.strip_suffix(".gltf") {
                                    Some(n) => n,
                                    None => ""
                                },
                            };
                            if name.contains(&state.search_str_gltf) && name != "" {
                                if ui.button(name) {
                                    self.state.change_model(file_name, &self.gpu, resource_files_directory.to_string(), default_texture_path)
                                };
                            }
                        };
                    });
                    ui.same_line();
                    ChildWindow::new("maps").build(&ui, || {
                        ui.input_text("map name", &mut state.save_map_name).build();
                        //ui.modal
                        if ui.button("Save map") {
                            if state.save_map_name != "" {
                                self.state.instances.save_temap(&state.save_map_name, map_files_directory.to_string())
                            } else {
                                ui.open_popup("SAVE FAILED");
                            }
                        };
                        if let Some(_token) = PopupModal::new("SAVE FAILED").begin_popup(&ui) {
                            ui.text("Please write the file name when saving");
                            if ui.button("OK") {
                                ui.close_current_popup();
                            }
                        }
                        ui.separator();
                        if ui.button("Refresh") {
                            self.maps = get_map_names(map_files_directory);
                        };
                        ui.input_text("search", &mut state.search_str_temap).build();
                        for file_name in &self.maps {
                            let name = match file_name.strip_suffix(".temap") {
                                Some(n) => n,
                                None => "",
                            };
                            if name.contains(&state.search_str_temap) && name != "" {
                                if ui.button(name) {
                                    self.state.load_map(file_name, &self.gpu, map_files_directory.to_string(), resource_files_directory.to_string(), default_texture_path);
                                    //self.state.calculate_render_matrix();
                                };
                            }
                        };
                    })
                });
            imgui::Window::new("Object control")
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(&ui, || {
                    let _state = &mut self.object_control_win;
                    let mod_inst = &mut self.state.instances.modifying_instance;
                    ui.text("position");
                    InputFloat::new(&ui, "x", &mut mod_inst.x).step(1.0).step_fast(5.0).build();
                    InputFloat::new(&ui, "y", &mut mod_inst.y).step(1.0).step_fast(5.0).build();
                    InputFloat::new(&ui, "z", &mut mod_inst.z).step(1.0).step_fast(5.0).build();
                    if ui.button("place") {
                        self.state.instances.place_model(&self.gpu.device, &self.gpu.queue, &self.state.texture_bind_group_layout, tile_size, resource_files_directory.to_string(), default_texture_path)
                    }
                    ui.separator();
                    ui.checkbox("Blink selected model", &mut self.state.blinking);
                    let mut blink_freq = self.state.blink_freq as i32;
                    Slider::new("Blinking frequency", 0, 20).build(&ui, &mut blink_freq);
                    self.state.blink_freq = blink_freq as u64;
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

    pub fn render(&mut self, window: &Window, tile_size: (f32, f32, f32), resource_files_directory: &str, map_files_directory: &str, default_texture_path: &str) -> Result<(), wgpu::SurfaceError> {
        let output = self.gpu.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.state.render(&view, &self.gpu, tile_size)?;
        self.render_imgui(&view, window, tile_size, resource_files_directory, map_files_directory, default_texture_path);
        output.present();

        Ok(())
    }
}

fn get_resource_names(resource_files_directory: &str) -> Vec<String> {
    let mut names = Vec::new();
    match std::fs::read_dir(std::path::Path::new(resource_files_directory)) {
        Ok(files) => for file in files {
            let file_name = file.unwrap().file_name().to_str().unwrap().to_string();
            if file_name.ends_with(".glb") || file_name.ends_with(".gltf") {
                names.push(file_name)
            }
        },
        Err(_) => names.push(format!("Error accessing the directory {}", resource_files_directory)),
    };

    names
}

fn get_map_names(map_files_directory: &str) -> Vec<String> {
    let mut names = Vec::new();
    match std::fs::read_dir(std::path::Path::new(map_files_directory)) {
        Ok(files) => for file in files {
            let file_name = file.unwrap().file_name().to_str().unwrap().to_string();
            if file_name.ends_with(".temap") {
                names.push(file_name)
            }
        },
        Err(_) => names.push(format!("Error accessing the directory {}", map_files_directory)),
    };

    names
}