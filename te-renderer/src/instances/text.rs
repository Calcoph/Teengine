use wgpu::util::DeviceExt;

use crate::model;

use super::{InstanceRaw, InstancedDraw, Instance2D, Instance};

#[derive(Debug)]
pub struct InstancedText {
    pub image: model::Material,
    pub instance: Instance2D,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
}

impl InstancedText {
    pub fn new(
        image: model::Material,
        device: &wgpu::Device,
        x: f32,
        y: f32,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let mut instance = Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
        };

        let instance_data = [instance.to_raw()];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedText {
            image,
            instance,
            instance_buffer,
            depth,
        }
    }

    pub fn resize<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        index: usize,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        self.instance.resize(new_size);
        let raw = self.instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn animate(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.instance_buffer,
            (0 * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[self.instance.to_raw()]),
        );
    }
}

impl InstancedDraw for InstancedText {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        self.instance.move_direction(direction);
        let raw = self.instance.to_raw();
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
        self.instance.move_to(position);
        let raw = self.instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }
}

/// Handle of a 2D text. You will need it when changing its properties.
pub struct TextReference {
    pub index: usize,
}
