use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::{window::Window, event::WindowEvent};
use wgpu;

use te_renderer::{state::{State, GpuState}, initial_config::InitialConfiguration, camera::SAFE_CAMERA_ANGLE};

pub struct ImguiState {
    gpu: GpuState,
    pub context: Context,
    pub platform: WinitPlatform,
    renderer: Renderer,
    pub state: State,
    resources: Directory,
    maps: Directory,
    camera_controls_win: CameraControlsWin,
    model_selector_win: ModelSelectorWin,
    object_control_win: ObjectControlWin
}

struct CameraControlsWin {
    show_hotkeys: bool,
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
        config: InitialConfiguration
    ) -> Self {
        let size = window.inner_size();
        let gpu = GpuState::new(size, window).await;
        let state = State::new(window, &gpu, config.clone()).await;
        let mut context = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut context);
        platform.attach_window(context.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);

        let renderer_config = RendererConfig {
            texture_format: gpu.config.format,
            ..Default::default()
        };

        let renderer = Renderer::new(&mut context, &gpu.device, &gpu.queue, renderer_config);

        let resources = get_resource_names(&config.resource_files_directory, "");

        let maps = get_map_names(&config.map_files_directory, "");

        let camera_controls_win = CameraControlsWin {
            show_hotkeys: false,
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
                    let camera = &mut self.state.camera;
                    let state = &mut self.camera_controls_win;

                    let mut speed = camera.get_speed();
                    Slider::new("speed", 0.0 as f32, 3000.0 as f32).build(&ui, &mut speed);
                    camera.set_speed(speed);

                    let mut sensitivity = camera.get_sensitivity();
                    Slider::new("sensitivity", 0.0 as f32, 20.0 as f32).build(&ui, &mut sensitivity);
                    camera.set_sensitivity(sensitivity);

                    let mut fovy = camera.get_fovy();
                    Slider::new("fovy", 0.1 as f32, 179.9 as f32).build(&ui, &mut fovy);
                    camera.set_fovy(fovy);

                    let mut znear = camera.get_znear();
                    Slider::new("znear", 1.0 as f32, 1000.0 as f32).build(&ui, &mut znear);
                    camera.set_znear(znear);

                    let mut zfar = camera.get_zfar();
                    Slider::new("zfar", 100.0 as f32, 100000.0 as f32).build(&ui, &mut zfar);
                    camera.set_zfar(zfar);

                    let mut yaw = camera.get_yaw();
                    Slider::new("yaw", -2.0*SAFE_CAMERA_ANGLE as f32, 2.0*SAFE_CAMERA_ANGLE as f32).build(&ui, &mut yaw);
                    camera.set_yaw(yaw);

                    let mut pitch = camera.get_pitch();
                    VerticalSlider::new("pitch", [50.0, 100.0], -SAFE_CAMERA_ANGLE as f32, SAFE_CAMERA_ANGLE as f32).build(&ui, &mut pitch);
                    camera.set_pitch(pitch);

                    ui.same_line();
                    let mut zoom = 0.0;
                    VerticalSlider::new("zooom", [20.0, 100.0], -4.0 as f32, 4.0 as f32).build(&ui, &mut zoom);
                    
                    ui.same_line();
                    if ui.button("Go to origin") {
                        camera.move_absolute((0.0, 0.0, 0.0))
                    }
                    camera.set_zoom(zoom);
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
                            self.resources = get_resource_names(resource_files_directory, "");
                        };
                        ui.input_text("search", &mut state.search_str_gltf).build();
                        show_resources_directory(resource_files_directory, &self.resources, &ui, state, &self.gpu, &mut self.state)
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
                            self.maps = get_map_names(map_files_directory, "");
                        };
                        ui.input_text("search", &mut state.search_str_temap).build();
                        show_maps_directory(map_files_directory, &self.maps, &ui, state, &self.gpu, &mut self.state);
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

fn show_resources_directory(root: &str, dir: &Directory, ui: &Ui, window_state: &ModelSelectorWin, gpu: &GpuState, state: &mut State) {
    ui.text(&dir.directory_name);
    ui.separator();

    for file in &dir.files {
        match file {
            File::F(file_name) => {
                let name = match file_name.strip_suffix(".glb") {
                    Some(n) => n,
                    None => match file_name.strip_suffix(".gltf") {
                        Some(n) => n,
                        None => ""
                    },
                };
                if name.contains(&window_state.search_str_gltf) && name != "" {
                    if ui.button(name) {
                        state.change_model(&(dir.directory_name.clone() + "/" + &file_name), gpu) // TODO: don't hardcode "/"
                    };
                }
            },
            File::D(nested_dir) => show_resources_directory(root, &nested_dir, ui, window_state, gpu, state),
        }
    }
}

fn show_maps_directory(root: &str, dir: &Directory, ui: &Ui, window_state: &ModelSelectorWin, gpu: &GpuState, state: &mut State) {
    ui.text(&dir.directory_name);
    ui.separator();

    for file in &dir.files {
        match file {
            File::F(file_name) => {
                let name = match file_name.strip_suffix(".temap") {
                    Some(n) => n,
                    None => "",
                };
                if name.contains(&window_state.search_str_temap) && name != "" {
                    if ui.button(name) {
                        state.load_map(&(dir.directory_name.clone() + "/" + &file_name), gpu); // TODO: don't hardcode "/"
                        //state.calculate_render_matrix();
                    };
                }
            },
            File::D(nested_dir) => show_maps_directory(root, &nested_dir, ui, window_state, gpu, state),
        }
    };

}

fn get_resource_names(resource_files_directory: &str, dir_name: &str) -> Directory {
    let mut dir = Directory { directory_name: dir_name.to_string(), files: Vec::new() };
    let path = resource_files_directory.to_string() + "/" + dir_name;
    let path = std::path::Path::new(&(path));
    match std::fs::read_dir(path) {
        Ok(files) => for file in files {
            let file_name = file.unwrap().file_name().to_str().unwrap().to_string();
            let name = match dir_name {
                "" => "".to_string(),
                s => s.to_string() + "/" // TODO: don't hardcode "/"
            };
            let full_path = name + &file_name;
            let full_path = std::path::Path::new(&(full_path));
            if path.join(full_path).is_dir() {
                dir.files.push(File::D(get_resource_names(resource_files_directory, full_path.to_str().unwrap())))
            } else if file_name.ends_with(".glb") || file_name.ends_with(".gltf") {
                dir.files.push(File::F(file_name));
            }
        },
        Err(_) => dir.files.push(File::F(format!("Error accessing the directory {}", dir_name))),
    };

    dir
}

fn get_map_names(map_files_directory: &str, dir_name: &str) -> Directory {
    let mut dir = Directory { directory_name: dir_name.to_string(), files: Vec::new() };
    let path = map_files_directory.to_string() + "/" + dir_name; // TODO: don't hardcode "/"
    let path = std::path::Path::new(&(path));
    match std::fs::read_dir(path) {
        Ok(files) => for file in files {
            let file_name = file.unwrap().file_name().to_str().unwrap().to_string();
            let name = match dir_name {
                "" => "".to_string(),
                s => s.to_string() + "/" // TODO: don't hardcode "/"
            };
            let full_path = name + &file_name;
            let full_path = std::path::Path::new(&(full_path));
            if path.join(full_path).is_dir() {
                dir.files.push(File::D(get_map_names(map_files_directory, full_path.to_str().unwrap())))
            } else if file_name.ends_with(".temap") {
                dir.files.push(File::F(file_name))
            }
        },
        Err(_) => dir.files.push(File::F(format!("Error accessing the directory {}", dir_name))),
    };

    dir
}

enum File {
    F(String),
    D(Directory)
}

struct Directory {
    directory_name: String,
    files: Vec<File>
}