use std::marker::PhantomData;

use wgpu::{Device, BufferDescriptor, BufferUsages, Queue, CommandEncoder, CommandEncoderDescriptor};

const RESIZING_FACTOR: u64 = 2;

pub trait BufferObject {
    fn as_data(&self) -> &[u8];
    fn data_size() -> u64;
}

pub struct GpuVec<T: BufferObject> {
    capacity: u64,
    length: u64,
    buffer: wgpu::Buffer,
    label: Option<String>,
    phantom_data: PhantomData<T>
}

impl<T: BufferObject> GpuVec<T> {
    pub fn new(device: &Device, label: Option<&str>, usage: BufferUsages) -> GpuVec<T> {
        Self::with_capacity(device, label, usage, 1)
    }

    pub fn with_capacity(device: &Device, label: Option<&str>, usage: BufferUsages, capacity: u64) -> GpuVec<T> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size: capacity * T::data_size(),
            usage,
            mapped_at_creation: false,
        });

        GpuVec {
            capacity,
            length: 0,
            buffer,
            label: label.map(|a| a.to_string()),
            phantom_data: PhantomData
        }
    }

    #[must_use]
    pub fn push(&mut self, elem: &T, queue: &Queue, device: &Device) -> Option<CommandEncoder> {
        if self.length < self.capacity {
            self.add(self.length, elem, queue);
            None
        } else {
            let encoder = self.resize(self.capacity * RESIZING_FACTOR, device);
            self.add(self.length, elem, queue);
            Some(encoder)
        }
    }

    fn add(&mut self, index: u64, elem: &T, queue: &Queue) {
        let offset = index * T::data_size();
        self.length += 1;

        queue.write_buffer(
            &self.buffer,
            offset,
            elem.as_data()
        )
    }

    #[must_use]
    fn resize(&mut self, new_size: u64, device: &Device) -> CommandEncoder {
        self.capacity = new_size;
        let mut new_buffer = device.create_buffer(&BufferDescriptor {
            label: self.label.as_ref().map(|a| a.as_str()),
            size: self.capacity * T::data_size(),
            usage: self.buffer.usage(),
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("GpuVec resizer") });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &new_buffer,
            0,
            self.buffer.size()
        );

        std::mem::swap(&mut self.buffer, &mut new_buffer);

        encoder
    }
}
