use std::borrow::Cow;

use cgmath::{Vector2, Point3, Point2, vec2, Vector3};
pub use glyph_brush::{Section, Text};
use wgpu::{
    util::DeviceExt, BindGroupLayout, CommandBuffer, CommandEncoder, InstanceDescriptor,
    PushConstantRange, ShaderStages, InstanceFlags, RenderPipeline,
};
use winit::{
    dpi,
    event::{WindowEvent, KeyEvent},
    window::Window,
};
// TODO: Tell everyone when screen is resized, so instances' in_viewport can be updated
pub use crate::instances::builders::*;
use crate::{
    camera::{self, CameraState},
    initial_config::InitialConfiguration,
    instances::{InstanceRaw, InstancesState, Opaque3DInstance},
    model, temap, texture,
};
#[allow(deprecated)]
use crate::{
    error::TError,
    instances::{animation::Animation, text::OldTextReference, InstanceReference},
    model::{Material, Vertex},
    render::{Draw2D, InstanceFinder, Renderer, RendererClickable},
    text::{FontError, FontReference, TextState},
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
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
            flags: InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic
        });
        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Unable to create surface")
        };
        // TODO: instance.enumerate_adapters to list all GPUs (tutorial 2 beginner)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), //TODO: add option to select between LowPower, HighPerformance or default
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Unable to request adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::PUSH_CONSTANTS.union(wgpu::Features::POLYGON_MODE_LINE), // TODO: Make this optional
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits {
                            max_push_constant_size: 4,
                            ..wgpu::Limits::downlevel_webgl2_defaults()
                        }
                    } else {
                        wgpu::Limits {
                            max_push_constant_size: 4,
                            ..wgpu::Limits::default()
                        }
                    },
                    label: None,
                },
                None,
            )
            .await
            .expect("Unable to request device");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface.get_capabilities(&adapter).formats[0]],
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TeColor {
    red: f64,
    green: f64,
    blue: f64,
}

impl TeColor {
    pub fn new(red: f64, green: f64, blue: f64) -> TeColor {
        let mut color = TeColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
        };

        color.set_red(red);
        color.set_green(green);
        color.set_blue(blue);

        color
    }

    pub fn get_red(&self) -> f64 {
        self.red
    }

    pub fn get_u8_red(&self) -> u8 {
        (self.red * u8::MAX as f64) as u8
    }

    pub fn set_red(&mut self, mut red: f64) {
        if red < 0.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 0.0"
            );
            red = 0.0;
        } else if red > 1.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 1.0"
            );
            red = 1.0;
        }
        self.red = red
    }

    pub fn get_green(&self) -> f64 {
        self.green
    }

    pub fn get_u8_green(&self) -> u8 {
        (self.green * u8::MAX as f64) as u8
    }

    pub fn set_green(&mut self, mut green: f64) {
        if green < 0.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 0.0"
            );
            green = 0.0;
        } else if green > 1.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 1.0"
            );
            green = 1.0;
        }
        self.green = green
    }

    pub fn get_blue(&self) -> f64 {
        self.blue
    }

    pub fn get_u8_blue(&self) -> u8 {
        (self.blue * u8::MAX as f64) as u8
    }

    pub fn set_blue(&mut self, mut blue: f64) {
        if blue < 0.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 0.0"
            );
            blue = 0.0;
        } else if blue > 1.0 {
            eprintln!(
                "TeColor values must be between 0.0 and 1.0. It was automatically set to 1.0"
            );
            blue = 1.0;
        }
        self.blue = blue
    }
}

struct BindGroupLayouts {
    texture: wgpu::BindGroupLayout,
    camera: wgpu::BindGroupLayout,
    projection: wgpu::BindGroupLayout,
}

impl BindGroupLayouts {
    fn new(device: &wgpu::Device) -> Self {
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

        BindGroupLayouts {
            texture: texture_bind_group_layout,
            camera: camera_bind_group_layout,
            projection: projection_bind_group_layout,
        }
    }
}

struct PipeLineLayouts {
    render_pipeline_layout: wgpu::PipelineLayout,
    sprite_render_pipeline_layout: wgpu::PipelineLayout,
    clickable_pipeline_layout: wgpu::PipelineLayout,
}

impl PipeLineLayouts {
    fn new(
        device: &wgpu::Device,
        bind_group_layouts: &BindGroupLayouts
    ) -> PipeLineLayouts {
        let BindGroupLayouts {
            texture: texture_bind_group_layout,
            camera: camera_bind_group_layout,
            projection: projection_bind_group_layout,
        } = bind_group_layouts;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sprite_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sprite Render Pipeline Layout"),
                bind_group_layouts: &[&projection_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let clickable_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clickable Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX,
                    range: 0..4,
                }],
            });

        PipeLineLayouts {
            render_pipeline_layout,
            sprite_render_pipeline_layout,
            clickable_pipeline_layout,
        }
    }
}

#[derive(Debug)]
pub struct PipeLines {
    pub render_3d: wgpu::RenderPipeline,
    pub wireframe: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub sprite: wgpu::RenderPipeline,
    pub clickable: wgpu::RenderPipeline,
    pub clickable_color: wgpu::RenderPipeline,
}

impl PipeLines {
    fn new(gpu: &GpuState, layouts: PipeLineLayouts) -> Self {
        let PipeLineLayouts {
            render_pipeline_layout,
            sprite_render_pipeline_layout,
            clickable_pipeline_layout,
        } = layouts;
        let shader_3d: Cow<_> = include_str!("shader.wgsl").into();
        let clickable_shader: Cow<_> = include_str!("clickable_shader.wgsl").into();
        let shader_2d: Cow<_> = include_str!("2d_shader.wgsl").into();

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_3D"),
                source: wgpu::ShaderSource::Wgsl(shader_3d.clone()),
            };
            create_render_pipeline(
                &gpu.device,
                &render_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                EntryPoint::Mask,
                "Render Pipeline",
            )
        };

        let wireframe_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_3D"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_wireframe_pipeline(
                &gpu.device,
                &render_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                EntryPoint::Mask,
                "Wireframe Render Pipeline",
            )
        };

        let transparent_render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_3D_transparent"),
                source: wgpu::ShaderSource::Wgsl(shader_3d),
            };
            create_render_pipeline(
                &gpu.device,
                &render_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                EntryPoint::Main,
                "Transparent pipeline",
            )
        };

        let clickable_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_clickable"),
                source: wgpu::ShaderSource::Wgsl(clickable_shader.clone()),
            };
            create_r32uint_render_pipeline(
                &gpu.device,
                &clickable_pipeline_layout,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                EntryPoint::Main,
                "Clickable pipeline",
            )
        };

        let clickable_color_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_clickable"),
                source: wgpu::ShaderSource::Wgsl(clickable_shader),
            };
            create_render_pipeline(
                &gpu.device,
                &clickable_pipeline_layout,
                gpu.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
                EntryPoint::Color,
                "Clickable color pipeline",
            )
        };

        let sprite_render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader_sprites"),
                source: wgpu::ShaderSource::Wgsl(shader_2d),
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

        PipeLines {
            render_3d: render_pipeline,
            wireframe: wireframe_pipeline,
            transparent: transparent_render_pipeline,
            sprite: sprite_render_pipeline,
            clickable: clickable_pipeline,
            clickable_color: clickable_color_pipeline,
        }
    }
}

#[derive(Debug)]
pub struct TeState {
    /// Manages the camera
    pub camera: camera::CameraState,
    /// The window's size
    pub size: winit::dpi::PhysicalSize<u32>,
    pub pipelines: PipeLines,
    /// Manages 3D models, 2D sprites and 2D texts
    pub instances: InstancesState,
    pub text: TextState,
    maps_path: String,
    sprite_vertices_buffer: wgpu::Buffer,
    /// Whether to render 3d models
    pub render_3d: bool,
    /// Whether to render 2D sprites and instanced texts.
    pub render_2d: bool,
    /// Whether to render texts.
    pub render_text: bool,
    pub bgcolor: TeColor,
    instance_finder: InstanceFinder,
}

impl TeState {
    pub async fn new(window: &Window, gpu: &GpuState, init_config: InitialConfiguration) -> Self {
        let size = window.inner_size();
        let maps_path = init_config.map_files_directory.clone();
        let resources_path = init_config.resource_files_directory.clone();
        let default_texture_path = init_config.default_texture_path.clone();

        let bindg_layouts = BindGroupLayouts::new(&gpu.device);

        let camera = CameraState::new(
            &gpu.config,
            &gpu.device,
            &bindg_layouts.camera,
            &bindg_layouts.projection,
            init_config.clone(),
        );

        let pipelines = PipeLines::new(gpu, PipeLineLayouts::new(&gpu.device, &bindg_layouts));

        let instances = InstancesState::new(
            bindg_layouts.texture,
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
        let text = TextState::new();
        TeState {
            camera,
            size,
            pipelines,
            instances,
            text,
            maps_path,
            sprite_vertices_buffer,
            render_2d: true,
            render_3d: true,
            render_text: true,
            bgcolor: TeColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
            },
            instance_finder: InstanceFinder::new(),
        }
    }

    pub fn load_font(
        &mut self,
        font_path: String,
        gpu: &GpuState,
    ) -> Result<FontReference, FontError> {
        self.text
            .load_font(font_path, &gpu.device, gpu.config.format)
    }

    pub fn load_map(&mut self, file_name: &str, gpu: &GpuState) -> Result<(), TError> {
        let map = temap::TeMap::from_file(file_name, self.maps_path.clone());
        self.instances.fill_from_temap(map, gpu)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.camera.resize(new_size);
    }

    // returns if the event has been fully processed
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key,
                        ..
                    },
                ..
            } => self.camera.camera_controller.process_keyboard(*physical_key, *state),
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
            gpu.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Text encoder"),
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

            for (_name, sprite) in &mut self.instances.sprite_instances.instanced {
                sprite.animate(queue);
            }
        }
    }

    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        gpu: &GpuState,
        encoders: &mut Vec<wgpu::CommandEncoder>,
        texts: &[(FontReference, Vec<Section>)],
    ) {
        if self.render_3d {
            let mut render_pass = encoders
                .get_mut(0)
                .expect("Empty encoders vector")
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
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gpu.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            self.draw_opaque(&mut render_pass, &self.pipelines.render_3d);
            self.draw_transparent(&mut render_pass, &self.pipelines.transparent);
        }

        if self.render_2d {
            let mut render_pass = encoders
                .get_mut(1)
                .expect("Encoders vector too small")
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[
                        // This is what [[location(0)]] in the fragment shader targets
                        Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            self.draw_sprites(&mut render_pass);
        }

        if self.render_text {
            self.draw_text(
                &gpu.device,
                encoders.get_mut(2).expect("Encoders vector too small"),
                view,
                texts,
            );
        }
    }

    pub fn render_wireframe(&mut self,
        view: &wgpu::TextureView,
        gpu: &GpuState,
        encoders: &mut Vec<wgpu::CommandEncoder>
    ) {
        if self.render_3d {
            let mut render_pass = encoders
                .get_mut(0)
                .expect("Empty encoders vector")
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("wireframe Render Pass"),
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
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gpu.depth_texture.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            self.draw_opaque(&mut render_pass, &self.pipelines.wireframe);
            self.draw_transparent(&mut render_pass, &self.pipelines.wireframe)
        }
    }

    pub fn end_render(&mut self, gpu: &GpuState, encoders: Vec<wgpu::CommandEncoder>) {
        self.text.end_render();
        let encoders: Vec<CommandBuffer> = encoders
            .into_iter()
            .map(|encoder| encoder.finish())
            .collect();

        gpu.queue.submit(encoders);
    }

    pub fn clicakble_mask(
        &mut self,
        view: &wgpu::TextureView,
        gpu: &GpuState,
        encoder: &mut wgpu::CommandEncoder,
        drawable: bool,
        depth_texture: Option<&wgpu::TextureView>,
    ) {
        let depth_texture = match depth_texture {
            Some(dt) => dt,
            None => &gpu.depth_texture.view,
        };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[
                // This is what [[location(0)]] in the fragment shader targets
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.draw_clickable(&mut render_pass, drawable);
    }

    fn draw_clickable<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>, drawable: bool) {
        if drawable {
            render_pass.set_pipeline(&self.pipelines.clickable_color);
        } else {
            render_pass.set_pipeline(&self.pipelines.clickable);
        }
        let mut renderer = RendererClickable::new(render_pass, &self.camera.camera_bind_group);

        let iter = self
            .instances
            .opaque_instances
            .instanced
            .iter()
            .filter(|(_name, instanced_model)| instanced_model.unculled_instances > 0);


        for (name, instanced_model) in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));

            renderer.draw_model_instanced_mask(
                &instanced_model.model,
                instanced_model.get_instances_vec(),
                name.to_owned(),
            );
        }

        let iter = self
            .instances
            .opaque_instances
            .animated
            .iter()
            .filter(|(_name, instanced_model)| instanced_model.unculled_instance);


        for (_name, instanced_model) in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            // TODO: pass name
            renderer.draw_animated_model_instanced_mask(&instanced_model);
        }

        // transparent
        let iter = self
            .instances
            .transparent_instances
            .iter()
            .map(|name| {
                self.instances
                    .opaque_instances
                    .instanced
                    .get(name)
                    .expect("Invalid reference")
            })
            .filter(|instanced_model| instanced_model.unculled_instances > 0);
        for instanced_model in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            renderer.tdraw_model_instanced_mask(&instanced_model.model, instanced_model.get_instances_vec());
        }

        // transparent
        let iter = self
            .instances
            .transparent_instances
            .iter()
            .map(|name| {
                self.instances
                    .opaque_instances
                    .animated
                    .get(name)
                    .expect("Invalid reference")
            })
            .filter(|instanced_model| instanced_model.unculled_instance);
        for instanced_model in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            renderer.tdraw_animated_model_instanced_mask(instanced_model);
        }

        self.instance_finder = renderer.get_instance_finder();
    }

    pub fn draw_opaque<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, pipeline: &'a RenderPipeline) {
        render_pass.set_pipeline(pipeline);
        let mut renderer = Renderer::new(render_pass, &self.camera.camera_bind_group);
        let iter = self
            .instances
            .opaque_instances
            .instanced
            .iter()
            .filter(|(_name, instanced_model)| instanced_model.unculled_instances > 0);
        for (_name, instanced_model) in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            renderer.draw_model_instanced(&instanced_model.model, instanced_model.get_instances_vec());
        }

        let iter = self
            .instances
            .opaque_instances
            .animated
            .iter()
            .filter(|(_name, instanced_model)| instanced_model.unculled_instance);
        for (_name, instanced_model) in iter {
            let instance_buffer = &instanced_model.instance_buffer;
            renderer
                .render_pass
                .set_vertex_buffer(1, instance_buffer.slice(..));
            renderer.draw_animated_model_instanced(instanced_model);
        }
    }

    pub fn draw_transparent<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, pipeline: &'a RenderPipeline) {
        render_pass.set_pipeline(pipeline);
        let mut renderer = Renderer::new(render_pass, &self.camera.camera_bind_group);

        let iter = self
            .instances
            .transparent_instances
            .iter()
            .map(|name| {
                self.instances.opaque_instances.get(name).unwrap()
            })
            .filter(|instanced_model| instanced_model.is_unculled());
        for instanced_model in iter {
            match instanced_model {
                Opaque3DInstance::Normal(instanced_model) => {
                    let instance_buffer = &instanced_model.instance_buffer;
                    renderer
                        .render_pass
                        .set_vertex_buffer(1, instance_buffer.slice(..));
                    renderer.tdraw_model_instanced(&instanced_model.model, instanced_model.get_instances_vec());
                },
                Opaque3DInstance::Animated(instanced_model) => {
                    let instance_buffer = &instanced_model.instance_buffer;
                    renderer
                        .render_pass
                        .set_vertex_buffer(1, instance_buffer.slice(..));
                    renderer.tdraw_animated_model_instanced(&instanced_model);
                }
            }
        };
    }

    pub fn draw_sprites<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipelines.sprite);
        let mut sorted_2d: Vec<&dyn Draw2D> = self
            .instances
            .texts
            .iter()
            .filter(|text| {
                // TODO: use filter_map
                if let Some(t) = text {
                    t.is_drawable()
                } else {
                    false
                }
            })
            .map(|text| text.as_ref().expect("Unreachable") as &dyn Draw2D)
            .collect();
        let sprites: Vec<&dyn Draw2D> = self
            .instances
            .sprite_instances
            .instanced
            .iter()
            .map(|(_name, inst)| inst as &dyn Draw2D)
            .collect();
        sorted_2d.extend(sprites.into_iter());
        let anim_sprites: Vec<&dyn Draw2D> = self
            .instances
            .sprite_instances
            .animated
            .iter()
            .filter(|(_name, inst)| inst.is_drawable())
            .map(|(_name, inst)| inst as &dyn Draw2D)
            .collect();
        sorted_2d.extend(anim_sprites.into_iter());
        sorted_2d.sort_by(|inst1, inst2| {
            inst1
                .get_depth()
                .partial_cmp(&inst2.get_depth())
                .expect("Speciall f64 values such as NaN not allowed for instance depth")
        });
        for draw in sorted_2d {
            draw.draw(
                render_pass,
                &self.camera.projection_bind_group,
                &self.sprite_vertices_buffer,
            )
        }
    }

    pub fn draw_text(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        view: &wgpu::TextureView,
        sections: &[(FontReference, Vec<Section>)],
    ) {
        self.text
            .draw(device, encoder, view, vec2(self.size.width, self.size.height), sections)
    }

    fn cull_all3d(&mut self) {
        for (_name, model) in self.instances.opaque_instances.instanced.iter_mut() {
            model.cull_all();
        }

        for (_name, model) in self.instances.opaque_instances.animated.iter_mut() {
            model.cull_all();
        }
    }

    pub fn find_clicked_instance(&mut self, num: u32) -> Option<InstanceReference> {
        self.instance_finder.find_instance(num)
    }
}

// Calls to self.instances' methods
impl TeState {
    pub fn get_layout(&self) -> &BindGroupLayout {
        &self.instances.layout
    }

    pub fn add_model<'state, 'gpu, 'a>(
        &'state mut self,
        model_name: &'a str,
        gpu: &'gpu GpuState,
        position: Point3<f32>,
    ) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder::new(self, model_name, gpu, position)
    }

    /// Creates a new 3D model at the specific position.
    /// ### Errors
    /// Will error if the model's file is not found
    #[deprecated]
    pub fn place_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
    ) -> Result<InstanceReference, TError> {
        self.instances.place_model(model_name, gpu, tile_position.into())
    }

    /// Places an already created model at the specific position.
    /// If that model has not been forgotten, you can place another with just its name, so model can be None
    /// ### Errors
    /// Will error if model is None and the model has been forgotten (or was never created)
    #[deprecated]
    pub fn place_custom_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: Option<model::Model>,
    ) -> Result<InstanceReference, TError> {
        self.instances
            .place_custom_model(model_name, gpu, tile_position.into(), model)
    }

    #[deprecated]
    pub fn place_custom_model_absolute(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: Option<model::Model>,
    ) -> Result<InstanceReference, TError> {
        self.instances
            .place_custom_model_absolute(model_name, gpu, tile_position.into(), model)
    }

    /// Places an already created animated model at the specific position.
    #[deprecated]
    pub fn place_custom_animated_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: model::AnimatedModel,
    ) -> InstanceReference {
        self.instances
            .place_custom_animated_model(model_name, gpu, tile_position.into(), model)
    }

    pub fn add_sprite<'a, 'b, 'c, 'd>(
        &'a mut self,
        sprite_name: &'b str,
        gpu: &'c GpuState,
        position: Point2<f32>,
        depth: f32
    ) -> SpriteBuilder<'a, 'c, 'b, 'd> {
        SpriteBuilder::new(self, sprite_name, gpu, position, depth)
    }

    /// Creates a new 2D sprite at the specified position.
    /// All 2D sprites created from the same file will have the same "z" position. And cannot be changed once set.
    /// ### Errors
    /// Will error if the sprite's file is not found
    #[deprecated]
    pub fn place_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<Vector2<f32>>,
        position: Point2<f32>,
        depth: f32,
        force_new_instance_id: Option<&str>,
    ) -> Result<InstanceReference, TError> {
        let screen_size = vec2(self.size.width, self.size.height);
        self.instances.place_sprite(
            sprite_name,
            gpu,
            size,
            position,
            depth,
            screen_size,
            force_new_instance_id,
        )
    }

    #[deprecated]
    pub fn place_custom_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<Vector2<f32>>,
        position: Point2<f32>,
        depth: f32,
        sprite: Option<(Material, Vector2<f32>)>,
    ) -> Result<InstanceReference, TError> {
        let screen_size = vec2(self.size.width, self.size.height);
        self.instances.place_custom_sprite(
            sprite_name,
            gpu,
            size,
            position,
            depth,
            screen_size,
            sprite,
        )
    }

    // TODO: Do this with builder
    pub fn place_animated_sprite(
        &mut self,
        sprite_names: Vec<&str>,
        gpu: &GpuState,
        size: Option<Vector2<f32>>,
        position: Point2<f32>,
        depth: f32,
        frame_delay: std::time::Duration,
        looping: bool,
    ) -> Result<InstanceReference, TError> {
        let screen_size = vec2(self.size.width, self.size.height);
        self.instances.place_animated_sprite(
            sprite_names,
            gpu,
            size,
            position,
            depth,
            frame_delay,
            looping,
            screen_size
        )
    }

    /// Creates a new text at the specified position
    /// ### PANICS
    /// will panic if the characters' files are not found
    /// see: model::Font
    #[deprecated]
    #[allow(deprecated)]
    pub fn place_old_text(
        &mut self,
        text: Vec<String>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
    ) -> OldTextReference {
        self.instances
            .place_text(text, gpu, size, position, self.size.width, self.size.height)
    }

    /// Eliminates the text from screen and memory.
    #[deprecated]
    #[allow(deprecated)]
    pub fn forget_old_text(&mut self, text: OldTextReference) {
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
    pub fn fill_from_temap(&mut self, map: temap::TeMap, gpu: &GpuState) -> Result<(), TError> {
        self.instances.fill_from_temap(map, gpu)
    }

    /// Move a 3D model or a 2D sprite relative to its current position.
    /// Ignores z value on 2D sprites.
    pub fn move_instance(
        &mut self,
        instance: &InstanceReference,
        direction: Vector3<f32>,
        queue: &wgpu::Queue,
    ) {
        let screen_size = vec2(self.size.width, self.size.height);
        self.instances.move_instance(
            instance,
            direction.into(),
            queue,
            screen_size
        )
    }

    /// Move a 3D model or a 2D sprite to an absolute position.
    /// Ignores z value on 2D sprites.
    pub fn set_instance_position(
        &mut self,
        instance: &InstanceReference,
        position: Point3<f32>,
        queue: &wgpu::Queue,
    ) {
        let screen_size = vec2(self.size.width, self.size.height);
        self.instances.set_instance_position(
            instance,
            position.into(),
            queue,
            screen_size
        )
    }

    /// Get a 3D model's or 2D sprite's position.
    pub fn get_instance_position(&self, instance: &InstanceReference) -> Point3<f32> {
        self.instances.get_instance_position(instance)
    }

    /// Changes the sprite's size. Using TODO algorithm
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn resize_sprite(
        &mut self,
        instance: &InstanceReference,
        new_size: Vector2<f32>,
        queue: &wgpu::Queue,
    ) {
        self.instances
            .resize_sprite(instance, new_size.into(), queue)
    }

    /// Get the sprite's size
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn get_sprite_size(&self, instance: &InstanceReference) -> Vector2<f32> {
        self.instances.get_sprite_size(instance)
    }

    /// Move a 2D text relative to it's current position.
    /// Ignores the z value.
    #[deprecated]
    #[allow(deprecated)]
    pub fn move_old_text<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &OldTextReference,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        self.instances.move_text(
            instance,
            direction.into(),
            queue,
            self.size.width,
            self.size.height,
        )
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn change_old_text_depth(
        &mut self,
        instance: &OldTextReference,
        depth: f32
    ) {
        self.instances.change_text_depth(
            instance,
            depth
        )
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn get_old_text_depth(
        &mut self,
        instance: &OldTextReference,
    ) -> f32 {
        self.instances.get_text_depth(
            instance
        )
    }

    /// Move a 2D text to an absolute position.
    /// Ignores the z value.
    #[deprecated]
    #[allow(deprecated)]
    pub fn set_old_text_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &OldTextReference,
        position: P,
        queue: &wgpu::Queue,
    ) {
        self.instances.set_text_position(
            instance,
            position.into(),
            queue,
            self.size.width,
            self.size.height,
        )
    }

    /// Gets a 2D text's position
    #[deprecated]
    #[allow(deprecated)]
    pub fn get_old_text_position(&self, instance: &OldTextReference) -> (f32, f32) {
        self.instances.get_text_position(instance)
    }

    /// Resizes a 2D text
    #[deprecated]
    #[allow(deprecated)]
    pub fn resize_old_text<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        instance: &OldTextReference,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        self.instances.resize_text(instance, new_size.into(), queue)
    }

    /// Gets a 2D text's size
    #[deprecated]
    #[allow(deprecated)]
    pub fn get_old_text_size(&self, instance: &OldTextReference) -> (f32, f32) {
        self.instances.get_text_size(instance)
    }

    pub fn set_instance_animation(&mut self, instance: &InstanceReference, animation: Animation) {
        self.instances.set_instance_animation(instance, animation)
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn set_old_text_animation(&mut self, text: &OldTextReference, animation: Animation) {
        self.instances.set_text_animation(text, animation)
    }

    pub fn animate_model(
        &mut self,
        instance: &InstanceReference,
        mesh_index: usize,
        material_index: usize,
    ) {
        self.instances
            .animate_model(instance, mesh_index, material_index)
    }

    pub fn hide_instance(&mut self, instance: &InstanceReference) {
        if !self.instances.is_instance_hidden(instance) {
            self.instances.hide_instance(instance)
        }
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn hide_old_text(&mut self, instance: &OldTextReference) {
        if !self.instances.is_text_hidden(instance) {
            self.instances.hide_text(instance)
        }
    }

    pub fn show_instance(&mut self, instance: &InstanceReference) {
        if self.instances.is_instance_hidden(instance) {
            self.instances.show_instance(instance)
        }
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn show_old_text(&mut self, instance: &OldTextReference) {
        if self.instances.is_text_hidden(instance) {
            self.instances.show_text(instance)
        }
    }

    pub fn is_hidden(&self, instance: &InstanceReference) -> bool {
        self.instances.is_instance_hidden(instance)
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn is_old_text_hidden(&self, instance: &OldTextReference) -> bool {
        self.instances.is_text_hidden(instance)
    }

    pub fn is_frustum_culled(&self, instance: &InstanceReference) -> bool {
        self.instances.is_frustum_culled(instance)
    }
}

enum EntryPoint {
    Main,
    Mask,
    Color,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    entry_point: EntryPoint,
    name: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let entry_point = match entry_point {
        EntryPoint::Main => "fs_main",
        EntryPoint::Mask => "fs_mask",
        EntryPoint::Color => "fs_color",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
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

fn create_wireframe_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    entry_point: EntryPoint,
    name: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let entry_point = match entry_point {
        EntryPoint::Main => "fs_main",
        EntryPoint::Mask => "fs_mask",
        EntryPoint::Color => "fs_color",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
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
                write_mask: wgpu::ColorWrites::COLOR,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,            
            polygon_mode: wgpu::PolygonMode::Line,
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

fn create_r32uint_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    entry_point: EntryPoint,
    name: &str,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    let entry_point = match entry_point {
        EntryPoint::Main => "fs_main",
        EntryPoint::Mask => "fs_mask",
        EntryPoint::Color => "fs_color",
    };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
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
                format: wgpu::TextureFormat::R32Uint,
                blend: None,
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
