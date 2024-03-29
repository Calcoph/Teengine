use std::cell::RefCell;

use cgmath::{Vector2, Point2, Vector3};
use wgpu::util::DeviceExt;

use crate::model;

use super::{rangetree::RangeTree, Instance2D, InstanceRaw};

#[derive(Debug)]
pub struct InstancedSprite {
    pub sprite: model::Material,
    size: Vector2<f32>,
    pub instances: Vec<Instance2D>,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
    shown_instances: RangeTree,
}

impl InstancedSprite {
    pub fn new(
        sprite: model::Material,
        device: &wgpu::Device,
        position: Point2<f32>,
        depth: f32,
        size: Vector2<f32>,
        screen_size: Vector2<u32>,
    ) -> Self {
        let instances = vec![Instance2D::new(
            position,
            size,
            None,
            screen_size
        )];

        InstancedSprite::new_premade(sprite, device, instances, depth, size)
    }

    fn new_premade(
        sprite: model::Material,
        device: &wgpu::Device,
        mut instances: Vec<Instance2D>,
        depth: f32,
        size: Vector2<f32>
    ) -> Self {
        let instance_data = instances
            .iter_mut()
            .map(Instance2D::to_raw)
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let mut shown_instances = RangeTree::new();
        if let Some(last) = instances.first() {
            if last.in_viewport {
                shown_instances.add_num(0)
            }
        }

        InstancedSprite {
            sprite,
            size,
            instances,
            instance_buffer,
            depth,
            shown_instances,
        }
    }

    pub(crate) fn add_instance(
        &mut self,
        position: Point2<f32>,
        size: Option<Vector2<f32>>,
        device: &wgpu::Device,
        screen_size: Vector2<u32>
    ) {
        let size = match size {
            Some(size) => size,
            None => self.size,
        };
        let new_instance = Instance2D::new(
            position,
            size,
            None,
            screen_size
        );
        //TODO: see if there is a better way than replacing the buffer with a new one
        self.instances.push(new_instance);
        if self.instances.last().expect("Unreachable").in_viewport {
            self.shown_instances
                .add_num(self.instances.len() as u32 - 1)
        }
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

    pub fn resize(
        &mut self,
        index: usize,
        new_size: Vector2<f32>,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        instance.resize(new_size);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
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
                        .expect("Too many instances"), // rare case when usize > u64
                    bytemuck::cast_slice(&[i.to_raw()]),
                );
            }
        }
    }

    pub(crate) fn show(&mut self, index: usize) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        if instance.is_hidden() {
            instance.show();
            if instance.in_viewport {
                self.shown_instances.add_num(index as u32)
            }
        }
    }

    pub(crate) fn hide(&mut self, index: usize) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        if !instance.is_hidden() {
            instance.hide();
            if instance.in_viewport {
                self.shown_instances.remove_num(index as u32)
            }
        }
    }

    pub(crate) fn is_hidden(&self, index: usize) -> bool {
        let instance = self.instances.get(index).expect("Unreachable");
        instance.is_hidden()
    }

    pub(crate) fn get_instances_vec(&self) -> Vec<std::ops::Range<u32>> {
        self.shown_instances.get_vec()
    }

    pub(crate) fn move_instance(
        &mut self,
        index: usize,
        direction: Vector3<f32>,
        queue: &wgpu::Queue,
        screen_size: Vector2<u32>
    ) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        if let Some(show) = instance.move_direction(direction, screen_size) {
            if show {
                self.shown_instances.add_num(index as u32)
            } else {
                self.shown_instances.remove_num(index as u32)
            }
        };
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn set_instance_position(
        &mut self,
        index: usize,
        position: Point2<f32>,
        queue: &wgpu::Queue,
        screen_size: Vector2<u32>
    ) {
        let instance = self.instances.get_mut(index).expect("Unreachable");
        if let Some(show) = instance.move_to(position, screen_size) {
            if show {
                self.shown_instances.add_num(index as u32)
            } else {
                self.shown_instances.remove_num(index as u32)
            }
        };
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

#[derive(Debug)]
pub struct AnimatedSprite {
    sprites: Vec<model::Material>,
    pub instance: Instance2D,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
    start_time: RefCell<std::time::Instant>,
    frame_delay: std::time::Duration,
    looping: bool,
}

impl AnimatedSprite {
    pub fn new(
        sprites: Vec<model::Material>,
        device: &wgpu::Device,
        position: Point2<f32>,
        depth: f32,
        size: Vector2<f32>,
        frame_delay: std::time::Duration,
        looping: bool,
        screen_size: Vector2<u32>
    ) -> Self {
        let mut instance = Instance2D::new(
            position,
            size,
            None,
            screen_size
        );

        let instance_data = instance.to_raw();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&[instance_data]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        AnimatedSprite {
            sprites,
            instance,
            instance_buffer,
            depth,
            start_time: RefCell::new(std::time::Instant::now()),
            frame_delay,
            looping,
        }
    }

    pub fn play_animation(&mut self) {
        self.start_time = RefCell::new(std::time::Instant::now())
    }

    pub fn get_sprite(&self) -> &model::Material {
        let now = std::time::Instant::now();
        self.get_sprite_rec(now)
    }

    fn get_sprite_rec(&self, now: std::time::Instant) -> &model::Material {
        let dt = now - *self.start_time.borrow();
        let frame = (dt.as_secs_f32() / self.frame_delay.as_secs_f32()).floor() as usize;
        match self.sprites.get(frame) {
            Some(frame) => frame,
            None => {
                if self.looping {
                    *self.start_time.borrow_mut() += self.frame_delay * self.sprites.len() as u32;
                    self.get_sprite_rec(now)
                } else {
                    self.sprites.last().expect("Unreachable")
                }
            }
        }
    }

    pub fn resize(
        &mut self,
        index: usize,
        new_size: Vector2<f32>,
        queue: &wgpu::Queue,
    ) {
        self.instance.resize(new_size);
        let raw = self.instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn hide(&mut self) {
        self.instance.hide();
    }

    pub(crate) fn show(&mut self) {
        self.instance.show();
    }

    pub(crate) fn is_hidden(&self) -> bool {
        self.instance.is_hidden()
    }

    pub(crate) fn move_instance(
        &mut self,
        index: usize,
        direction: Vector3<f32>,
        queue: &wgpu::Queue,
        screen_size: Vector2<u32>
    ) {
        let _ = self.instance.move_direction(direction, screen_size);
        let raw = self.instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn set_instance_position(
        &mut self,
        index: usize,
        position: Point2<f32>,
        queue: &wgpu::Queue,
        screen_size: Vector2<u32>
    ) {
        let _ = self.instance.move_to(position, screen_size);
        let raw = self.instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .expect("Too many instances"), // rare case when usize > u64
            bytemuck::cast_slice(&[raw]),
        );
    }

    pub(crate) fn is_drawable(&self) -> bool {
        self.instance.in_viewport && !self.instance.hidden
    }
}
