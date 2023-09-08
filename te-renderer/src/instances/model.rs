use std::ops::Range;

use wgpu::util::DeviceExt;

use super::{rangetree::RangeTree, Instance, Instance3D, InstanceRaw, InstancedDraw};
use crate::model;

#[derive(Debug)]
pub struct InstancedModel {
    pub model: model::Model,
    pub instances: Vec<Instance3D>,
    pub instance_buffer: wgpu::Buffer,
    pub unculled_instances: usize,
    unculled_indices: RangeTree,
}

impl InstancedModel {
    pub fn new(model: model::Model, device: &wgpu::Device, x: f32, y: f32, z: f32) -> Self {
        let instances = vec![Instance3D {
            position: cgmath::Vector3 { x, y, z },
            animation: None,
            hidden: false,
        }];

        InstancedModel::new_premade(model, device, instances)
    }

    pub fn new_premade(
        model: model::Model,
        device: &wgpu::Device,
        mut instances: Vec<Instance3D>,
    ) -> Self {
        let instance_data = instances
            .iter_mut()
            .map(Instance3D::to_raw)
            .collect::<Vec<_>>();
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
            unculled_indices: RangeTree::new(),
        }
    }

    pub fn add_instance(&mut self, x: f32, y: f32, z: f32, device: &wgpu::Device) {
        let new_instance = Instance3D {
            position: cgmath::Vector3 { x, y, z },
            animation: None,
            hidden: false,
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

    pub(crate) fn uncull_instance(&mut self, index: usize) {
        if !self.unculled_indices.contains(&(index as u32)) {
            if !self.instances.get(index).expect("Unreachable").hidden {
                self.unculled_instances += 1;
                self.unculled_indices.add_num(index as u32);
            }
        }
    }

    pub(crate) fn cull_all(&mut self) {
        self.unculled_instances = 0;
        self.unculled_indices.clean();
    }

    pub(crate) fn get_instances_vec(&self) -> Vec<Range<u32>> {
        self.unculled_indices.get_vec()
    }

    pub(crate) fn hide(&mut self, index: usize) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        instance.hide()
    }

    pub(crate) fn show(&mut self, index: usize) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        instance.show()
    }

    pub(crate) fn is_hidden(&self, index: usize) -> bool {
        let instance = self.instances.get(index).expect("Unreachable");
        instance.is_hidden()
    }
}

impl InstancedDraw for InstancedModel {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        instance.move_direction(direction);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }

    fn set_instance_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        position: P,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        instance.move_to(position);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }
}
