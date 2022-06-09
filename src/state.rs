use std::collections::HashMap;

use winit::{window::Window, event::{WindowEvent, KeyboardInput, ElementState}, dpi};
use wgpu::util::DeviceExt;
use cgmath::prelude::*;

use crate::{texture, camera, resources::load_glb_model};
use crate::{model, model::Vertex};
use crate::resources;
use crate::consts as c;
use crate::temap;

/* const NUM_INSTANCES_PER_ROW: u32 = 1;
const INSTANCE_DISPLACEMENT: cgmath::Vector3<f32> =
    cgmath::Vector3::new(
        NUM_INSTANCES_PER_ROW as f32 * 0.5,
        0.0,
        NUM_INSTANCES_PER_ROW as f32 * 0.5
    ); */

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into()
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        // We're using Vector4 because of the uniforms 16 byte spacing requirement
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4
                }
            ]
        }
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model = cgmath::Matrix4::from_translation(self.position);
        InstanceRaw {
            model: model.into(),
        }
    }
}

pub struct CameraState {
    camera: camera::Camera,
    pub projection: camera::Projection,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    pub camera_controller: camera::CameraController,
    camera_bind_group: wgpu::BindGroup,
}

impl CameraState {
    fn new(config: &wgpu::SurfaceConfiguration, device: &wgpu::Device, camera_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let camera = camera::Camera::new(
            c::CAMERA_START_POSITION,
            cgmath::Deg(c::CAMERA_START_YAW),
            cgmath::Deg(c::CAMERA_START_PITCH)
        );
        let projection = camera::Projection::new(
            config.width,
            config.height,
            cgmath::Deg(c::FOVY),
            c::ZNEAR,
            c::ZFAR
        );
        let camera_controller =
            camera::CameraController::new(
                c::CAMERA_SPEED,
                c::CAMERA_SENSITIVITY
            );
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding()
                }
            ],
            label: Some("camera_bind_group")
        });

        CameraState {
            camera,
            projection,
            camera_uniform,
            camera_buffer,
            camera_controller,
            camera_bind_group,
        }
    }

    fn update(&mut self, dt: std::time::Duration, queue: &wgpu::Queue) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.projection.resize(new_size.width, new_size.height);
    }
}

pub struct GpuState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    depth_texture: texture::Texture,
}

impl GpuState {
    pub async fn new(size: dpi::PhysicalSize<u32>, window: &Window) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        // TODO: instance.enumerate_adapters to list all GPUs (tutorial 2 beginner)
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(), //TODO: add option to select between LowPower, HighPerformance or default
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None
            },
            None
        ).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode:wgpu::PresentMode::Fifo 
        };
        surface.configure(&device, &config);

        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        GpuState {
            surface,
            device,
            queue,
            config,
            depth_texture
        }
    }

    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }
}
struct InstancedModel {
    model: model::Model,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

impl InstancedModel {
    fn new(model: model::Model, device: &wgpu::Device, x: f32, y: f32, z: f32) -> Self {
        let instances = vec![Instance {
            position: cgmath::Vector3 { x, y, z },
        }];

        InstancedModel::new_premade(model, device, instances)
    }

    fn new_premade(model: model::Model, device: &wgpu::Device, instances: Vec<Instance>) -> Self {
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        let instances = instances.into_iter().map(|instance| {
            instance
        }).collect();

        InstancedModel {
            model,
            instances,
            instance_buffer
        }
    }

    fn add_instance(&mut self, x: f32, y: f32, z: f32, device: &wgpu::Device) {
        let new_instance = Instance {
            position: cgmath::Vector3 { x, y, z },
        };
        //TODO: see if there is a better way than replacing the buffer with a new one
        self.instances.push(new_instance);
        self.instance_buffer.destroy();
        let instance_data = self.instances.iter().map(|instance| instance.to_raw()).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
            }
        );
        self.instance_buffer = instance_buffer;
    }
}

pub struct ModifyingInstance {
    model: model::Model,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    buffer: Option<wgpu::Buffer>
}

impl ModifyingInstance {
    fn into_renderable(&mut self, device: &wgpu::Device) -> usize {
        let instances = vec![Instance {
            position: cgmath::Vector3 { x: self.x*c::TILE_SIZE, y: self.y*c::TILE_SIZE, z: self.z*c::TILE_HEIGHT },
        }];

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX
            }
        );
        self.buffer = Some(instance_buffer);
        instances.len()
    }
}

pub struct InstancesState {
    instances: HashMap<String, InstancedModel>,
    modifying_name: String,
    pub modifying_instance: ModifyingInstance,
    
}

impl InstancesState {
    fn new(gpu: &GpuState, texture_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let instances = HashMap::new();
        let modifying_name = "box02.glb".to_string();
        let model = resources::load_glb_model(
            "box02.glb", 
            &gpu.device, 
            &gpu.queue,
            texture_bind_group_layout,
        ).unwrap();
        let modifying_instance = ModifyingInstance {
            model,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            buffer: None
        };
        InstancesState {
            instances,
            modifying_name,
            modifying_instance,
        }
    }

    fn set_model(&mut self, file_name: &str, model: model::Model) {
        self.modifying_name = file_name.to_string();
        self.modifying_instance.model = model
    }

    pub fn place_model(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout) {
        match self.instances.contains_key(&self.modifying_name) {
            true => {
                let x = self.modifying_instance.x * c::TILE_SIZE;
                let y = self.modifying_instance.y * c::TILE_SIZE;
                let z = self.modifying_instance.z * c::TILE_HEIGHT;
                let instanced_m = self.instances.get_mut(&mut self.modifying_name).unwrap();
                instanced_m.add_instance(x, y, z, device);
            },
            false => {
                let model = load_glb_model(&self.modifying_name, device, queue, layout).unwrap(); // TODO: clone the model instead of reading it again
                let x = self.modifying_instance.x * c::TILE_SIZE;
                let y = self.modifying_instance.y * c::TILE_SIZE;
                let z = self.modifying_instance.z * c::TILE_HEIGHT;
                let instanced_m = InstancedModel::new(model, device, x, y, z);
                self.instances.insert(self.modifying_name.clone(), instanced_m);
            },
        }
    }

    pub fn save_temap(&self, file_name: &str) {
        let mut map = temap::TeMap::new();
        for (name, instance) in &self.instances {
            map.add_model(&name);
            for inst in &instance.instances {
                map.add_instance(inst.position.x, inst.position.y, inst.position.z)
            }
        }

        map.save(&file_name);
    }

    fn from_temap(map: temap::TeMap, gpu: &GpuState, texture_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let mut instances_state = InstancesState::new(gpu, texture_bind_group_layout);
        for (name, te_model) in map.models {
            let model = load_glb_model(&name, &gpu.device, &gpu.queue, texture_bind_group_layout);
            match model {
                Ok(m) => {
                    let mut instances = Vec::new();
                    for offset in te_model.offsets {
                        let instance = Instance {
                            position: cgmath::Vector3 {
                                x: offset.x,
                                y: offset.y,
                                z: offset.z
                            },
                        };
                        instances.push(instance);
                    };
                    instances_state.instances.insert(name, InstancedModel::new_premade(m, &gpu.device, instances));
                },
                _ => ()
            }
        };

        instances_state
    }
}

pub struct State {
    pub camera: CameraState,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    transparent_render_pipeline: wgpu::RenderPipeline,
    pub instances: InstancesState,
    pub mouse_pressed: bool,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub blinking: bool,
    blink_time: std::time::Instant,
    pub blink_freq: u64,
}

impl State {
    pub async fn new(window: &Window, gpu: &GpuState) -> Self {
        let size = window.inner_size();
        
        let (texture_bind_group_layout,
            camera_bind_group_layout,
            render_pipeline_layout
        ) = State::get_layouts(&gpu.device);
        
        let camera = CameraState::new(&gpu.config, &gpu.device, &camera_bind_group_layout);

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
                false
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
                true
            )
        };

        let instances = InstancesState::new(gpu, &texture_bind_group_layout);

        let blinking = true;
        let blink_time = std::time::Instant::now();
        let blink_freq = 1;

        State {
            camera,
            size,
            render_pipeline,
            transparent_render_pipeline,
            instances,
            mouse_pressed: false,
            texture_bind_group_layout,
            blinking,
            blink_time,
            blink_freq,
        }
    }

    pub fn load_map(&mut self, file_name: &str, gpu: &GpuState) {
        let map = temap::TeMap::from_file(file_name);
        self.instances = InstancesState::from_temap(map, gpu, &self.texture_bind_group_layout);
    }

    fn get_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None
                    }
                ],
                label: Some("texture_bind_group_layout")
            });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                }
            ],
            label: Some("camera_bind_group_layout")
        });

        let render_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[]
            }
        );

        (texture_bind_group_layout, camera_bind_group_layout, render_pipeline_layout)
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
            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            },
            _ => false
        }
    }

    pub fn update(&mut self, dt: std::time::Duration, gpu: &GpuState) {
        self.camera.update(dt, &gpu.queue);
    }

    pub fn render(&mut self, view: &wgpu::TextureView, gpu: &GpuState) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.6,
                                b: 0.2,
                                a: 1.0
                            }),
                            store: true
                        }
                    }
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

            
            use model::DrawModel;
            render_pass.set_pipeline(&self.render_pipeline);
            for (_name, instanced_model) in self.instances.instances.iter() {
                render_pass.set_vertex_buffer(1, instanced_model.instance_buffer.slice(..));
                render_pass.draw_model_instanced(
                    &instanced_model.model,
                    0..instanced_model.instances.len() as u32,
                    &self.camera.camera_bind_group,
                );
            }
            let time_elapsed = std::time::Instant::now() - self.blink_time;
            let model_visible = !self.blinking || time_elapsed < std::time::Duration::new(self.blink_freq, 0);
            let mut instances = 0;
            let mut buffer = None;
            let mut model = None;
            if model_visible {
                // TODO: make a function instead of copy-pasting for modifying_instance
                instances = self.instances.modifying_instance.into_renderable(&gpu.device);
                buffer = self.instances.modifying_instance.buffer.as_ref();
                model = Some(&self.instances.modifying_instance.model);
                render_pass.set_vertex_buffer(1, buffer.unwrap().slice(..));
                render_pass.draw_model_instanced(
                    &model.unwrap(),
                    0..instances as u32,
                    &self.camera.camera_bind_group,
                );
            // 1 second = 1_000_000_000 nanoseconds
            // 500_000_000ns = 1/2 seconds
            } else if time_elapsed > std::time::Duration::new(self.blink_freq, 0)+std::time::Duration::new(0, 500_000_000) {
                self.blink_time = std::time::Instant::now();
            }


            use model::DrawTransparentModel;
            render_pass.set_pipeline(&self.transparent_render_pipeline);
            for (_name, instanced_model) in self.instances.instances.iter()
                .filter(|(_name, instanced_model)| {
                    instanced_model.model.transparent_meshes.len() > 0
                })
            {
                if instanced_model.model.transparent_meshes.len() > 0 {
                    render_pass.set_vertex_buffer(1, instanced_model.instance_buffer.slice(..));
                    render_pass.tdraw_model_instanced(
                        &instanced_model.model,
                        0..instanced_model.instances.len() as u32,
                        &self.camera.camera_bind_group,
                    );
                }
            }
            // TODO: make a function instead of copy-pasting for modifying_instance
            if model_visible {
                if model.unwrap().transparent_meshes.len() > 0 {
                    render_pass.set_vertex_buffer(1, buffer.unwrap().slice(..));
                    render_pass.tdraw_model_instanced(
                        &model.unwrap(),
                        0..instances as u32,
                        &self.camera.camera_bind_group,
                    );
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    pub fn change_model(&mut self, file_name: &str, gpu: &GpuState) {
        match resources::load_glb_model(
            file_name, 
            &gpu.device, 
            &gpu.queue,
            &self.texture_bind_group_layout,
        ) {
            Ok(model) => self.instances.set_model(file_name, model),
            Err(_) => (),
        };
    }
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
    transparent: bool
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
            cull_mode: None,//Some(wgpu::Face::Back),
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
        multiview: None
    })
}
