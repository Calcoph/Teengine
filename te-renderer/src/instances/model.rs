use wgpu::util::DeviceExt;

use crate::model;

use super::{InstancedDraw, InstanceRaw, Instance3D, Instance};


#[derive(Debug)]
pub struct InstancedModel {
    pub model: model::Model,
    pub instances: Vec<Instance3D>,
    pub instance_buffer: wgpu::Buffer,
    pub unculled_instances: usize,
    unculled_indices: Vec<usize>
}

impl InstancedModel {
    pub fn new(model: model::Model, device: &wgpu::Device, x: f32, y: f32, z: f32, chunk_size: (f32, f32, f32)) -> Self {
        let instances = vec![Instance3D {
            position: cgmath::Vector3 { x, y, z },
            animation: None,
        }];

        InstancedModel::new_premade(model, device, instances)
    }

    pub fn new_premade(model: model::Model, device: &wgpu::Device, mut instances: Vec<Instance3D>) -> Self {
        let instance_data = instances.iter_mut().map(Instance3D::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedModel {
            model,
            instances,
            instance_buffer,
            unculled_instances: 0,
            unculled_indices: Vec::new()
        }
    }

    pub fn add_instance(&mut self, x: f32, y: f32, z: f32, device: &wgpu::Device, chunk_size: (f32, f32, f32)) {
        let new_instance = Instance3D {
            position: cgmath::Vector3 { x, y, z },
            animation: None,
        };
        //TODO: see if there is a better way than replacing the buffer with a new one
        self.instances.push(new_instance);
        self.instance_buffer.destroy();
        let instance_data = self
            .instances
            .iter_mut()
            .map(|instance| instance.to_raw())
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.instance_buffer = instance_buffer;
    }

    pub(crate) fn uncull_instance(&mut self, queue: &wgpu::Queue, index: usize) {
        if !self.unculled_indices.contains(&index) {
            queue.write_buffer(
                &self.instance_buffer,
                (self.unculled_instances * std::mem::size_of::<InstanceRaw>())
                    .try_into()
                    .unwrap(),
                bytemuck::cast_slice(&[self.instances.get_mut(index).unwrap().to_raw()]),
            );
            self.unculled_instances += 1;
            self.unculled_indices.push(index);
        }
    }

    pub(crate) fn cull_all(&mut self) {
        self.unculled_instances = 0;
        self.unculled_indices = Vec::new();
    }
}

impl InstancedDraw for InstancedModel {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_direction(direction);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }

    fn set_instance_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        position: P,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_to(position);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }
}
