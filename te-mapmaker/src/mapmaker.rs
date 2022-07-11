use imgui_winit_support::WinitPlatform;
use imgui::*;
use imgui_wgpu::{Renderer, RendererConfig};
use winit::{window::Window, event::WindowEvent};
use wgpu;

use te_renderer::{state::{State, GpuState}, initial_config::InitialConfiguration, camera::SAFE_CAMERA_ANGLE, resources};

use crate::modifiying_instance::{self, InstancesState, ModifyingInstance};

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
    object_control_win: ObjectControlWin,
    mod_instance: modifiying_instance::InstancesState,
    renderer_s: RendererState
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
        config: InitialConfiguration,
        default_model: &str
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

        let blinking = true;
        let blink_time = std::time::Instant::now();
        let blink_freq = 1;
        let renderer_s = RendererState { blinking, blink_time, blink_freq };

        let mod_instance = InstancesState::new(&gpu,
            &state.instances.layout,
            state.instances.resources_path.clone(),
            &state.instances.default_texture_path,
            default_model);

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
            object_control_win,
            mod_instance,
            renderer_s
        }
    }

    pub fn render_imgui(&mut self, view: &wgpu::TextureView, window: &Window, resource_files_directory: &str, map_files_directory: &str) {
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
                    let (x, y, z) = camera.get_position();
                    ui.text(format!("position: {}x, {}y, {}z", x, y, z));
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
                        show_resources_directory(
                            resource_files_directory,
                            &self.resources,
                            &ui,
                            state,
                            &self.gpu,
                            &self.state,
                            &mut self.mod_instance,
                            &mut self.renderer_s
                        )
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
                        show_maps_directory(map_files_directory, &self.maps, &ui, state, &mut self.state, &self.gpu);
                    })
                });
            imgui::Window::new("Object control")
                .size([400.0, 200.0], Condition::FirstUseEver)
                .position([400.0, 200.0], Condition::FirstUseEver)
                .build(&ui, || {
                    let _state = &mut self.object_control_win;
                    let mod_inst = &mut self.mod_instance.modifying_instance;
                    ui.text("position");
                    InputFloat::new(&ui, "x", &mut mod_inst.x).step(1.0).step_fast(5.0).build();
                    InputFloat::new(&ui, "y", &mut mod_inst.y).step(1.0).step_fast(5.0).build();
                    InputFloat::new(&ui, "z", &mut mod_inst.z).step(1.0).step_fast(5.0).build();
                    if ui.button("place") {
                        let x = self.mod_instance.modifying_instance.x;
                        let y = self.mod_instance.modifying_instance.y;
                        let z = self.mod_instance.modifying_instance.z;
                        self.state.instances.place_model(
                            &self.mod_instance.modifying_name,
                            &self.gpu,
                            (x, y, z)
                            
                        );
                    }
                    ui.separator();
                    ui.checkbox("Blink selected model", &mut self.renderer_s.blinking);
                    let mut blink_freq = self.renderer_s.blink_freq as i32;
                    Slider::new("Blinking frequency", 0, 20).build(&ui, &mut blink_freq);
                    self.renderer_s.blink_freq = blink_freq as u64;
                });
            ui.show_demo_window(&mut opened);
        }

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ImGui Render Encoder")
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
            self.renderer.render(ui.render(), &self.gpu.queue, &self.gpu.device, &mut render_pass).expect("Rendering failed");
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish())); // TODO: call submit only once. right now it is called twice
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

    pub fn render(&mut self, window: &Window, tile_size: (f32, f32, f32), resource_files_directory: &str, map_files_directory: &str) -> Result<(), wgpu::SurfaceError> {
        let output = self.gpu.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer_s.render_state(&view, tile_size, &mut self.state, &self.gpu, &mut self.mod_instance.modifying_instance);
        self.render_imgui(&view, window, resource_files_directory, map_files_directory);
        output.present();

        Ok(())
    }

}

fn show_resources_directory(root: &str, dir: &Directory, ui: &Ui, window_state: &ModelSelectorWin, gpu: &GpuState, state: &State, mod_instance: &mut InstancesState, renderer_s: &mut RendererState) {
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
                        renderer_s.change_model(&(dir.directory_name.clone() + "/" + &file_name), gpu, state, mod_instance) // TODO: don't hardcode "/"
                    };
                }
            },
            File::D(nested_dir) => show_resources_directory(root, &nested_dir, ui, window_state, gpu, state, mod_instance, renderer_s),
        }
    }
}

fn show_maps_directory(root: &str, dir: &Directory, ui: &Ui, window_state: &ModelSelectorWin, state: &mut State, gpu: &GpuState) {
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
                        state.instances.forget_all_instances();
                        state.load_map(&(dir.directory_name.clone() + "/" + &file_name), &gpu); // TODO: don't hardcode "/"
                    };
                }
            },
            File::D(nested_dir) => show_maps_directory(root, &nested_dir, ui, window_state, state, gpu),
        }
    };
}

struct RendererState {
    blinking: bool,
    blink_time: std::time::Instant,
    blink_freq: u64
}

impl RendererState {
    fn render_state(&mut self, view: &wgpu::TextureView, tile_size: (f32, f32, f32), state: &mut State, gpu: &GpuState, modifying_instance: &mut ModifyingInstance) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });
        {    
            use te_renderer::model::DrawModel;
            let time_elapsed = std::time::Instant::now() - self.blink_time;
            let model_visible = !self.blinking || time_elapsed < std::time::Duration::new(self.blink_freq, 0);
            let mut instances = 0;
            let mut buffer = None;
            let mut model = None;
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0
                                }),
                                store: true
                            }
                        })
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gpu.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: true
                        }),
                        stencil_ops: None
                    })
                });
                state.draw_opaque(&mut render_pass);
                if model_visible {
                    instances = modifying_instance.into_renderable(&gpu.device, tile_size);
                    buffer = modifying_instance.buffer.as_ref();
                    model = Some(&modifying_instance.model);
                    render_pass.set_vertex_buffer(1, buffer.unwrap().slice(..));
                    render_pass.draw_model_instanced(
                        &model.unwrap(),
                        0..instances as u32,
                        &state.camera.camera_bind_group,
                    );
                // 1 second = 1_000_000_000 nanoseconds
                // 500_000_000ns = 1/2 seconds
                } else if time_elapsed > std::time::Duration::new(self.blink_freq, 0)+std::time::Duration::new(0, 500_000_000) {
                    self.blink_time = std::time::Instant::now();
                }
                state.draw_transparent(&mut render_pass);
                use te_renderer::model::DrawTransparentModel;
                if model_visible {
                    if model.unwrap().transparent_meshes.len() > 0 {
                        render_pass.set_vertex_buffer(1, buffer.unwrap().slice(..));
                        render_pass.tdraw_model_instanced(
                            &model.unwrap(),
                            0..instances as u32,
                            &state.camera.camera_bind_group,
                        );
                    }
                }
            }
        }
    
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn change_model(&mut self, file_name: &str, gpu: &GpuState, state: &State, mod_instance: &mut InstancesState) {
        match resources::load_glb_model(
            file_name, 
            &gpu.device, 
            &gpu.queue,
            &state.instances.layout,
            state.instances.resources_path.clone(),
            &state.instances.default_texture_path
        ) {
            Ok(model) => mod_instance.set_model(file_name, model),
            Err(_) => (),
        };
    }
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
            let file_path = std::path::Path::new(&(file_name));
            let full_path = name + &file_name;
            let full_path = std::path::Path::new(&(full_path));
            if path.join(file_path).is_dir() {
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
