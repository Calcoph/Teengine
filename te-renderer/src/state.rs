use wgpu::{util::DeviceExt, CommandBuffer, BindGroupLayout};
use winit::{
    dpi,
    event::{KeyboardInput, WindowEvent},
    window::Window,
};
// TODO: Tell everyone when screen is resized, so instances' in_viewport can be updated
use crate::{model::{Vertex, Material}, render::{DrawModel, Draw2D, DrawTransparentModel}, instances::{InstanceReference, text::TextReference, animation::Animation}};
use crate::{
    camera,
    initial_config::InitialConfiguration,
    instances::{InstanceRaw, InstancesState},
    model,
    temap, texture,
};

#[derive(Debug)]
pub struct GpuState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_texture: texture::Texture,
}

impl GpuState {
    pub async fn new(size: dpi::PhysicalSize<u32>, window: &Window) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        // TODO: instance.enumerate_adapters to list all GPUs (tutorial 2 beginner)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), //TODO: add option to select between LowPower, HighPerformance or default
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            //alpha_mode: wgpu::CompositeAlphaMode::Auto
        };
        surface.configure(&device, &config);

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        GpuState {
            surface,
            device,
            queue,
            config,
            depth_texture,
        }
    }

    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }
}

#[derive(Debug)]
pub struct TeColor {
    red: f64,
    green: f64,
    blue: f64
}

impl TeColor {
    pub fn get_red(&self) -> f64 {
        self.red
    }

    pub fn set_red(&mut self, mut red: f64) {
        if red < 0.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 0.0");
            red = 0.0;
        } else if red > 1.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 1.0");
            red = 1.0;
        }
        self.red = red
    }

    pub fn get_green(&self) -> f64 {
        self.green
    }

    pub fn set_green(&mut self, mut green: f64) {
        if green < 0.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 0.0");
            green = 0.0;
        } else if green > 1.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 1.0");
            green = 1.0;
        }
        self.green = green
    }

    pub fn get_blue(&self) -> f64 {
        self.blue
    }

    pub fn set_blue(&mut self, mut blue: f64) {
        if blue < 0.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 0.0");
            blue = 0.0;
        } else if blue > 1.0 {
            eprintln!("TeClor values must be between 0.0 and 1.0. It was automatically set to 1.0");
            blue = 1.0;
        }
        self.blue = blue
    }
}

#[derive(Debug)]
pub struct TeState {
    /// Manages the camera
    pub camera: camera::CameraState,
    /// The window's size
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    transparent_render_pipeline: wgpu::RenderPipeline,
    sprite_render_pipeline: wgpu::RenderPipeline,
    /// Manages 3D models, 2D sprites and 2D texts
    pub instances: InstancesState,
    maps_path: String,
    sprite_vertices_buffer: wgpu::Buffer,
    /// Whether to render 3d models
    pub render_3d: bool,
    /// Whether to render 2D sprites and texts.
    pub render_2d: bool,
    pub bgcolor: TeColor
}

impl TeState {
    pub async fn new(window: &Window, gpu: &GpuState, init_config: InitialConfiguration) -> Self {
        let size = window.inner_size();
        let maps_path = init_config.map_files_directory.clone();
        let resources_path = init_config.resource_files_directory.clone();
        let default_texture_path = init_config.default_texture_path.clone();

        let (
            texture_bind_group_layout,
            camera_bind_group_layout,
            render_pipeline_layout,
            projection_bind_group_layout,
            sprite_render_pipeline_layout,
        ) = TeState::get_layouts(&gpu.device);

        let camera = camera::CameraState::new(
            &gpu.config,
            &gpu.device,
            &camera_bind_group_layout,
            &projection_bind_group_layout,
            init_config.clone(),
        );

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &gpu.device,
                &render_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                false,
            )
        };

        let transparent_render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &gpu.device,
                &render_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                true,
            )
        };

        let sprite_render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("2d_shader.wgsl").into()),
            };
            create_2d_render_pipeline(
                &gpu.device,
                &sprite_render_pipeline_layout,
                gpu.config.format,
                &[model::SpriteVertex::desc(), InstanceRaw::desc()],
                shader,
                true,
            )
        };

        let instances = InstancesState::new(
            texture_bind_group_layout,
            init_config.tile_size,
            init_config.chunk_size,
            resources_path,
            default_texture_path,
            init_config.font_dir_path,
        );

        let sprite_vertices = &[
            model::SpriteVertex {
                position: [0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            model::SpriteVertex {
                position: [1.0, 0.0],
                tex_coords: [1.0, 0.0],
            },
            model::SpriteVertex {
                position: [0.0, 0.0],
                tex_coords: [0.0, 0.0],
            },
            model::SpriteVertex {
                position: [0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            model::SpriteVertex {
                position: [1.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
            model::SpriteVertex {
                position: [1.0, 0.0],
                tex_coords: [1.0, 0.0],
            },
        ];
        let sprite_vertices_buffer =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Sprite Vertex Buffer"),
                    contents: bytemuck::cast_slice(sprite_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        //instances.place_custom_model("Frustum", gpu, (0.0, 0.0, 0.0), Some(camera.get_frustum_model(gpu, &instances.layout)));
        TeState {
            camera,
            size,
            render_pipeline,
            transparent_render_pipeline,
            sprite_render_pipeline,
            instances,
            maps_path,
            sprite_vertices_buffer,
            render_2d: true,
            render_3d: true,
            bgcolor: TeColor { red: 0.0, green: 0.0, blue: 0.0 }
        }
    }

    pub fn load_map(&mut self, file_name: &str, gpu: &GpuState) {
        let map = temap::TeMap::from_file(file_name, self.maps_path.clone());
        self.instances.fill_from_temap(map, gpu);
    }

    fn get_layouts(
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::PipelineLayout,
        wgpu::BindGroupLayout,
        wgpu::PipelineLayout,
    ) {
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let projection_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("projection_bind_group_layout"),
            });

        let sprite_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sprite Render Pipeline Layout"),
                bind_group_layouts: &[&projection_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        (
            texture_bind_group_layout,
            camera_bind_group_layout,
            render_pipeline_layout,
            projection_bind_group_layout,
            sprite_render_pipeline_layout,
        )
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.camera.resize(new_size);
    }

    // returns if the event has been fully processed
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(key),
                        state,
                        ..
                    },
                ..
            } => self.camera.camera_controller.process_keyboard(*key, *state),
            _ => false,
        }
    }

    pub fn update(&mut self, dt: std::time::Duration, gpu: &GpuState) {
        self.animate(&gpu.queue);
        if self.render_3d {
            self.camera.update(dt, &gpu.queue);
            self.cull_all3d();
            self.instances.update_rendered3d(&self.camera.frustum);
        }
    }

    pub fn prepare_render(gpu: &GpuState) -> Vec<wgpu::CommandEncoder> {
        vec![
            gpu.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                }),
            gpu.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("2D Render Encoder"),
                }),
        ]
    }

    pub fn animate(&mut self, queue: &wgpu::Queue) {
        if self.render_2d {
            for text in &mut self.instances.texts {
                if let Some(txt) = text.as_mut() {
                    txt.animate(queue);
                }
            }
    
            for (_name, sprite) in &mut self.instances.sprite_instances {
                sprite.animate(queue);
            }
        }
    }

    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        gpu: &GpuState,
        encoders: &mut Vec<wgpu::CommandEncoder>,
    ) {
        if self.render_3d {
            let mut render_pass =
                encoders
                    .get_mut(0)
                    .unwrap()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[
                            // This is what [[location(0)]] in the fragment shader targets
                            Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: self.bgcolor.red,
                                        g: self.bgcolor.green,
                                        b: self.bgcolor.blue,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &gpu.depth_texture.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });
            self.draw_opaque(&mut render_pass);
            self.draw_transparent(&mut render_pass);
        }

        if self.render_2d {
            let mut render_pass =
                encoders
                    .get_mut(1)
                    .unwrap()
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Render Pass"),
                        color_attachments: &[
                            // This is what [[location(0)]] in the fragment shader targets
                            Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: true,
                                },
                            }),
                        ],
                        depth_stencil_attachment: None,
                    });
            self.draw_sprites(&mut render_pass);
        }
    }

    pub fn end_render(gpu: &GpuState, encoders: Vec<wgpu::CommandEncoder>) {
        let encoders: Vec<CommandBuffer> = encoders
            .into_iter()
            .map(|encoder| encoder.finish())
            .collect();
        gpu.queue.submit(encoders);
    }

    pub fn draw_opaque<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.render_pipeline);
        let iter = self.instances.opaque_instances
            .iter()
            .filter(|(_name, instanced_model)| {
                match instanced_model {
                    crate::instances::DrawModel::M(m) => m.unculled_instances > 0,
                    crate::instances::DrawModel::A(a) => a.unculled_instance,
                }
            });
        for (_name, instanced_model) in iter {
            let instance_buffer = match instanced_model {
                crate::instances::DrawModel::M(m) => &m.instance_buffer,
                crate::instances::DrawModel::A(a) => &a.instance_buffer,
            };
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
            match instanced_model {
                crate::instances::DrawModel::M(m) => {
                    render_pass.draw_model_instanced(
                        &m.model,
                        m.get_instances_vec(),
                        &self.camera.camera_bind_group,
                    );
                },
                crate::instances::DrawModel::A(a) => {
                    render_pass.draw_animated_model_instanced(
                        &a,
                        &self.camera.camera_bind_group,
                    );
                },
            }
        }
    }

    pub fn draw_transparent<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.transparent_render_pipeline);
        let iter = self
            .instances
            .transparent_instances
            .iter()
            .map(|name| self.instances.opaque_instances.get(name).unwrap())
            .filter(|instanced_model| {
                match instanced_model {
                    crate::instances::DrawModel::M(m) => m.unculled_instances > 0,
                    crate::instances::DrawModel::A(a) => a.unculled_instance,
                }
            });
        for instanced_model in iter {
            let instance_buffer = match instanced_model {
                crate::instances::DrawModel::M(m) => &m.instance_buffer,
                crate::instances::DrawModel::A(a) => &a.instance_buffer,
            };
            render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
            match instanced_model {
                crate::instances::DrawModel::M(m) => {
                    render_pass.tdraw_model_instanced(
                        &m.model,
                        m.get_instances_vec(),
                        &self.camera.camera_bind_group,
                    );
                },
                crate::instances::DrawModel::A(a) => {
                    render_pass.tdraw_animated_model_instanced(
                        &a,
                        &self.camera.camera_bind_group,
                    );
                },
            }
        }
    }

    pub fn draw_sprites<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.sprite_render_pipeline);
        let mut sorted_2d: Vec<&dyn Draw2D> = self
            .instances
            .texts
            .iter()
            .filter(|text| if let Some(t) = text {t.is_drawable()} else {false})
            .map(|text| text.as_ref().unwrap() as &dyn Draw2D)
            .collect();
        let sprites: Vec<&dyn Draw2D> = self.instances.sprite_instances.iter().map(|(_name, inst)| inst as &dyn Draw2D).collect();
        sorted_2d.extend(sprites.into_iter());
        let anim_sprites: Vec<&dyn Draw2D> = self.instances.animated_sprites.iter().filter(|(_name, inst)| inst.is_drawable()).map(|(_name, inst)| inst as &dyn Draw2D).collect();
        sorted_2d.extend(anim_sprites.into_iter());
        sorted_2d.sort_by(|inst1, inst2| inst1.get_depth().partial_cmp(&inst2.get_depth()).unwrap());
        for draw in sorted_2d {
            draw.draw(render_pass, &self.camera.projection_bind_group, &self.sprite_vertices_buffer)
        }
    }

    fn cull_all3d(&mut self) {
        for (_name, model) in self.instances.opaque_instances.iter_mut() {
            match model {
                crate::instances::DrawModel::M(m) => m.cull_all(),
                crate::instances::DrawModel::A(a) => a.cull_all(),
            }
        }
    }
/* 
    pub fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &InstanceReference,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        self.instances.move_instance(instance, direction.into(), queue, self.size.width, self.size.height)
    } */
}

// Calls to self.instances' methods
impl TeState {
    pub fn get_layout(&self) -> &BindGroupLayout {
        &self.instances.layout
    }

    /// Creates a new 3D model at the specific position.
    /// ### PANICS
    /// Will panic if the model's file is not found
    pub fn place_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
    ) -> InstanceReference {
        self.instances.place_model(model_name, gpu, tile_position)
    }

    /// Places an already created model at the specific position.
    /// If that model has not been forgotten, you can place another with just its name, so model can be None
    /// ### PANICS
    /// Will panic if model is None and the model has been forgotten (or was never created)
    pub fn place_custom_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: Option<model::Model>
    ) -> InstanceReference {
        self.instances.place_custom_model(model_name, gpu, tile_position, model)
    }

    pub fn place_custom_model_absolute(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: Option<model::Model>
    ) -> InstanceReference {
        self.instances.place_custom_model_absolute(model_name, gpu, tile_position, model)
    }

    /// Places an already created animated model at the specific position.
    pub fn place_custom_animated_model (
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: model::AnimatedModel
    ) -> InstanceReference {
        self.instances.place_custom_animated_model(model_name, gpu, tile_position, model)
    }

    /// Creates a new 2D sprite at the specified position.
    /// All 2D sprites created from the same file will have the same "z" position. And cannot be changed once set.
    /// ### PANICS
    /// Will panic if the sprite's file is not found
    pub fn place_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32)
    ) -> InstanceReference {
        self.instances.place_sprite(sprite_name, gpu, size, position, self.size.width, self.size.height)
    }

    pub fn place_custom_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        sprite: Option<(Material, f32, f32)>
    ) -> InstanceReference {
        self.instances.place_custom_sprite(sprite_name, gpu, size, position, self.size.width, self.size.height, sprite)
    }

    pub fn place_animated_sprite(
        &mut self,
        sprite_names: Vec<&str>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        frame_delay: std::time::Duration,
        looping: bool
    ) -> InstanceReference {
        self.instances.place_animated_sprite(sprite_names, gpu, size, position, frame_delay, looping, self.size.width, self.size.height)
    }

    /// Creates a new text at the specified position
    /// ### PANICS
    /// will panic if the characters' files are not found
    /// see: model::Font
    pub fn place_text(
        &mut self,
        text: Vec<String>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32)
    ) -> TextReference {
        self.instances.place_text(text, gpu, size, position, self.size.width, self.size.height)
    }

    /// Eliminates the text from screen and memory.
    pub fn forget_text(&mut self, text: TextReference) {
        self.instances.forget_text(text)
    }

    pub fn forget_all_2d_instances(&mut self) {
        self.instances.forget_all_2d_instances()
    }

    pub fn forget_all_3d_instances(&mut self) {
        self.instances.forget_all_3d_instances()
    }

    pub fn forget_all_instances(&mut self) {
        self.instances.forget_all_instances()
    }

    /// Saves all the 3D models' positions in a .temap file.
    pub fn save_temap(&self, file_name: &str, maps_path: String) {
        self.instances.save_temap(file_name, maps_path)
    }

    /// Load all 3D models from a .temap file.
    pub fn fill_from_temap(&mut self, map: temap::TeMap, gpu: &GpuState) {
        self.instances.fill_from_temap(map, gpu)
    }

    /// Move a 3D model or a 2D sprite relative to its current position.
    /// Ignores z value on 2D sprites.
    pub fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &InstanceReference,
        direction: V,
        queue: &wgpu::Queue
    ) {
        self.instances.move_instance(instance, direction.into(), queue, self.size.width, self.size.height)
    }

    /// Move a 3D model or a 2D sprite to an absolute position.
    /// Ignores z value on 2D sprites.
    pub fn set_instance_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &InstanceReference,
        position: P,
        queue: &wgpu::Queue
    ) {
        self.instances.set_instance_position(instance, position.into(), queue, self.size.width, self.size.height)
    }

    /// Get a 3D model's or 2D sprite's position.
    pub fn get_instance_position(&self, instance: &InstanceReference) -> (f32, f32, f32) {
        self.instances.get_instance_position(instance)
    }

    /// Changes the sprite's size. Using TODO algorithm
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn resize_sprite<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        instance: &InstanceReference,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        self.instances.resize_sprite(instance, new_size.into(), queue)
    }

    /// Get the sprite's size
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn get_sprite_size(&self, instance: &InstanceReference) -> (f32, f32) {
        self.instances.get_sprite_size(instance)
    }

    /// Move a 2D text relative to it's current position.
    /// Ignores the z value.
    pub fn move_text<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &TextReference,
        direction: V,
        queue: &wgpu::Queue
    ) {
        self.instances.move_text(instance, direction.into(), queue, self.size.width, self.size.height)
    }

    /// Move a 2D text to an absolute position.
    /// Ignores the z value.
    pub fn set_text_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &TextReference,
        position: P,
        queue: &wgpu::Queue,
    ) {
        self.instances.set_text_position(instance, position.into(), queue, self.size.width, self.size.height)
    }

    /// Gets a 2D text's position
    pub fn get_text_position(&self, instance: &TextReference) -> (f32, f32) {
        self.instances.get_text_position(instance)
    }

    /// Resizes a 2D text
    pub fn resize_text<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        instance: &TextReference,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        self.instances.resize_text(instance, new_size.into(), queue)
    }

    /// Gets a 2D text's size
    pub fn get_text_size(&self, instance: &TextReference) -> (f32, f32) {
        self.instances.get_text_size(instance)
    }

    pub fn set_instance_animation(&mut self, instance: &InstanceReference, animation: Animation) {
        self.instances.set_instance_animation(instance, animation)
    }

    pub fn set_text_animation(&mut self, text: &TextReference, animation: Animation) {
        self.instances.set_text_animation(text, animation)
    }

    pub fn animate_model(&mut self, instance: &InstanceReference, mesh_index: usize, material_index: usize) {
        self.instances.animate_model(instance, mesh_index, material_index)
    }

    pub fn hide_instance(&mut self, instance: &InstanceReference) {
        self.instances.hide_instance(instance)
    }

    pub fn hide_text(&mut self, instance: &TextReference) {
        self.instances.hide_text(instance)
    }

    pub fn show_instance(&mut self, instance: &InstanceReference) {
        self.instances.show_instance(instance)
    }

    pub fn show_text(&mut self, instance: &TextReference) {
        self.instances.show_text(instance)
    }

    pub fn is_hidden(&self, instance: &InstanceReference) -> bool {
        self.instances.is_instance_hidden(instance)
    }
    
    pub fn is_text_hidden(&self, instance: &TextReference) -> bool {
        self.instances.is_text_hidden(instance)
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    transparent: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let entry_point = match transparent {
        true => "fs_main",
        false => "fs_mask",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point,
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

fn create_2d_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    transparent: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let entry_point = match transparent {
        true => "fs_main",
        false => "fs_mask",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point,
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}
