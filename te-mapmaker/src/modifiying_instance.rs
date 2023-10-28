use cgmath::{point3, Point3, Vector3};
use te_renderer::{instances::Instance3D, model, resources, state::GpuState};
use wgpu::util::DeviceExt;
pub struct InstancesState {
    pub modifying_name: String,
    pub modifying_instance: ModifyingInstance,
}

impl InstancesState {
    pub fn new(
        gpu: &GpuState,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        resources_path: String,
        default_texture_path: &str,
        default_model: &str,
    ) -> Self {
        let modifying_name = default_model.to_string();
        let model = resources::load_glb_model(
            default_model,
            &gpu.device,
            &gpu.queue,
            texture_bind_group_layout,
            resources_path,
            default_texture_path,
        )
        .expect("Make sure the default texture and default models are valid");
        let modifying_instance = ModifyingInstance {
            model,
            position: point3(0.0, 0.0, 0.0),
            buffer: None,
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
    pub position: Point3<f32>,
    pub buffer: Option<wgpu::Buffer>,
}

impl ModifyingInstance {
    pub fn into_renderable(&mut self, device: &wgpu::Device, tile_size: Vector3<f32>) -> usize {
        let mut instances = vec![Instance3D {
            position: point3(
                self.position.x * tile_size.x,
                self.position.y * tile_size.y,
                self.position.z * tile_size.z,
            ),
            animation: None,
            hidden: false,
        }];

        use te_renderer::instances::Instance;
        let instance_data = instances
            .iter_mut()
            .map(Instance3D::to_raw)
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.buffer = Some(instance_buffer);
        instances.len()
    }
}
