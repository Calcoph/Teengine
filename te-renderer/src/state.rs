
use std::{cell::RefCell, collections::HashSet};

use wgpu::{util::DeviceExt, CommandBuffer};
use winit::{
    dpi,
    event::{KeyboardInput, WindowEvent},
    window::Window,
};

use crate::{model::Vertex, instances::{Instance3D, InstanceReference}};
use crate::{
    camera,
    initial_config::InitialConfiguration,
    instances::{InstanceRaw, sprite::InstancedSprite, text::InstancedText, InstancesState},
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
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
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
pub struct State {
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
}

impl State {
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
        ) = State::get_layouts(&gpu.device);

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
                Some(texture::Texture::DEPTH_FORMAT),
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

        State {
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
            self.cull_all();
            self.instances.update_rendered(&self.camera.frustum, &gpu.queue);
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
    
            for (_name, sprite) in &mut self.instances.instances_2d {
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
                            wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.0,
                                        g: 0.0,
                                        b: 0.0,
                                        a: 1.0,
                                    }),
                                    store: true,
                                },
                            },
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
            self.draw_opaque(&mut render_pass, &gpu.queue);
            self.draw_transparent(&mut render_pass, &gpu.queue);
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
                            wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: true,
                                },
                            },
                        ],
                        depth_stencil_attachment: None,
                    });
            self.draw_sprites(&mut render_pass, &gpu.queue);
        }
    }

    pub fn end_render(gpu: &GpuState, encoders: Vec<wgpu::CommandEncoder>) {
        let encoders: Vec<CommandBuffer> = encoders
            .into_iter()
            .map(|encoder| encoder.finish())
            .collect();
        gpu.queue.submit(encoders);
    }

    pub fn draw_opaque<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, queue: &wgpu::Queue) {
        use model::DrawModel;
        render_pass.set_pipeline(&self.render_pipeline);
        for (_name, instanced_model) in self.instances.instances.iter().filter(|(_name, instanced_model)| instanced_model.unculled_instances > 0) {
            render_pass.set_vertex_buffer(1, instanced_model.instance_buffer.slice(..));
            render_pass.draw_model_instanced(
                &instanced_model.model,
                0..instanced_model.unculled_instances as u32,
                &self.camera.camera_bind_group,
            );
        }
    }

    pub fn draw_transparent<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, queue: &wgpu::Queue) {
        use model::DrawTransparentModel;
        render_pass.set_pipeline(&self.transparent_render_pipeline);
        for (_name, instanced_model) in self
            .instances
            .instances
            .iter()
            .filter(|(_name, instanced_model)| instanced_model.unculled_instances > 0)
            .filter(|(_name, instanced_model)| instanced_model.model.transparent_meshes.len() > 0)
        {
            render_pass.set_vertex_buffer(1, instanced_model.instance_buffer.slice(..));
            render_pass.tdraw_model_instanced(
                &instanced_model.model,
                0..instanced_model.unculled_instances as u32,
                &self.camera.camera_bind_group,
            );
        }
    }

    pub fn draw_sprites<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, queue: &wgpu::Queue) {
        use model::DrawSprite;
        use model::DrawText;
        render_pass.set_pipeline(&self.sprite_render_pipeline);
        let mut sorted_texts: Vec<&InstancedText> = self
            .instances
            .texts
            .iter()
            .filter(|text| text.is_some())
            .map(|text| text.as_ref().unwrap())
            .collect();
        sorted_texts.sort_by(|inst1, inst2| inst1.depth.partial_cmp(&inst2.depth).unwrap());
        let mut sorted_sprites: Vec<(&String, &InstancedSprite)> =
            self.instances.instances_2d.iter().collect();
        sorted_sprites
            .sort_by(|(_n1, inst1), (_n2, inst2)| inst1.depth.partial_cmp(&inst2.depth).unwrap());
        let mut index_sprite = 0;
        let mut index_text = 0;
        loop {
            match sorted_sprites.get(index_sprite) {
                Some((_name, instanced_sprite)) => match sorted_texts.get(index_text) {
                    Some(instanced_text) => match instanced_text
                        .depth
                        .partial_cmp(&instanced_sprite.depth)
                        .unwrap()
                    {
                        std::cmp::Ordering::Less => {
                            render_pass
                                .set_vertex_buffer(1, instanced_text.instance_buffer.slice(..));
                            render_pass.draw_text(
                                &instanced_text.image,
                                &self.camera.projection_bind_group,
                                &self.sprite_vertices_buffer,
                            );
                            index_text += 1;
                        }
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
                            render_pass
                                .set_vertex_buffer(1, instanced_sprite.instance_buffer.slice(..));
                            render_pass.draw_sprite_instanced(
                                &instanced_sprite.sprite,
                                0..instanced_sprite.instances.len() as u32,
                                &self.camera.projection_bind_group,
                                &self.sprite_vertices_buffer,
                            );
                            index_sprite += 1;
                        }
                    },
                    None => {
                        render_pass
                            .set_vertex_buffer(1, instanced_sprite.instance_buffer.slice(..));
                        render_pass.draw_sprite_instanced(
                            &instanced_sprite.sprite,
                            0..instanced_sprite.instances.len() as u32,
                            &self.camera.projection_bind_group,
                            &self.sprite_vertices_buffer,
                        );
                        index_sprite += 1;
                    }
                },
                None => match sorted_texts.get(index_text) {
                    Some(instanced_text) => {
                        render_pass.set_vertex_buffer(1, instanced_text.instance_buffer.slice(..));
                        render_pass.draw_text(
                            &instanced_text.image,
                            &self.camera.projection_bind_group,
                            &self.sprite_vertices_buffer,
                        );
                        index_text += 1;
                    }
                    None => break,
                },
            }
        }
    }

    fn cull_all(&mut self) {
        for (_name, model) in self.instances.instances.iter_mut() {
            model.cull_all();
        }
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
    let shader = device.create_shader_module(&shader);
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
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }],
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
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    transparent: bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);
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
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }],
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
