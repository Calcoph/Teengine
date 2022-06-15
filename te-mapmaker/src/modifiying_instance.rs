use te_renderer::{resources, state::{GpuState, Instance3D}, model};
use wgpu::util::DeviceExt;
pub struct InstancesState {
    pub modifying_name: String,
    pub modifying_instance: ModifyingInstance,
}

impl InstancesState {
    pub fn new(gpu: &GpuState, texture_bind_group_layout: &wgpu::BindGroupLayout, resources_path: String, default_texture_path: &str, default_model: &str) -> Self {
        let modifying_name = default_model.to_string();
        let model = resources::load_glb_model(
            default_model, 
            &gpu.device, 
            &gpu.queue,
            texture_bind_group_layout,
            resources_path,
            default_texture_path
        ).unwrap();
        let modifying_instance = ModifyingInstance {
            model,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            buffer: None
        };
        InstancesState {
            modifying_name,
            modifying_instance,
        }
    }

    pub fn set_model(&mut self, file_name: &str, model: model::Model) {
        self.modifying_name = file_name.to_string();
        self.modifying_instance.model = model
    }
}

pub struct ModifyingInstance {
    pub model: model::Model,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub buffer: Option<wgpu::Buffer>
}

impl ModifyingInstance {
    pub fn into_renderable(&mut self, device: &wgpu::Device, tile_size: (f32, f32, f32)) -> usize {
        let instances = vec![Instance3D {
            position: cgmath::Vector3 { x: self.x*tile_size.0, y: self.y*tile_size.1, z: self.z*tile_size.2 },
        }];

        use te_renderer::state::Instance;
        let instance_data = instances.iter().map(Instance3D::to_raw).collect::<Vec<_>>();
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
