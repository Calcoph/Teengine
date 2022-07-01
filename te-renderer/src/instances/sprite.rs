use wgpu::util::DeviceExt;

use crate::model;

use super::{Instance2D, InstanceRaw, InstancedDraw, Instance};

#[derive(Debug)]
pub struct InstancedSprite {
    pub sprite: model::Material,
    width: f32,
    height: f32,
    pub instances: Vec<Instance2D>,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
}

impl InstancedSprite {
    pub fn new(
        sprite: model::Material,
        device: &wgpu::Device,
        x: f32,
        y: f32,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let instances = vec![Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
        }];

        InstancedSprite::new_premade(sprite, device, instances, depth, w, h)
    }

    fn new_premade(
        sprite: model::Material,
        device: &wgpu::Device,
        mut instances: Vec<Instance2D>,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let instance_data = instances.iter_mut().map(Instance2D::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedSprite {
            sprite,
            width: w,
            height: h,
            instances,
            instance_buffer,
            depth,
        }
    }

    pub fn add_instance(&mut self, x: f32, y: f32, size: Option<(f32, f32)>, device: &wgpu::Device) {
        let (w, h) = match size {
            Some((width, height)) => (width, height),
            None => (self.width, self.height),
        };
        let new_instance = Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
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

    pub fn resize<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        index: usize,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.resize(new_size);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn animate(&mut self, queue: &wgpu::Queue) {
        for (index, i) in self.instances.iter_mut().enumerate() {
            if i.animation.is_some() {
                queue.write_buffer(
                    &self.instance_buffer,
                    (index * std::mem::size_of::<InstanceRaw>())
                        .try_into()
                        .unwrap(),
                    bytemuck::cast_slice(&[i.to_raw()]),
                );
            }
        }
    }
}

impl InstancedDraw for InstancedSprite {
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
