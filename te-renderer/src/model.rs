use image::{ImageBuffer,Rgba};

use crate::state::GpuState;
use crate::texture::Texture;
use crate::{texture, instances};

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex for ModelVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteVertex {
    pub position: [f32; 2],
    pub tex_coords: [f32; 2],
}

impl Vertex for SpriteVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<SpriteVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        diffuse_texture: texture::Texture,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture,
            bind_group,
        }
    }
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub material: usize,
}

impl Mesh {
    pub fn new(name: String, model_name: &str, vertices: Vec<ModelVertex>, indices: Vec<u32>, material: usize, device: &wgpu::Device) -> Mesh {
        if vertices.len() == 0 {
            eprintln!("{model_name}");
            panic!("HAS 0 VERTICES")
        }
        let mut max_x = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        vertices
            .iter()
            .map(|vertex| (vertex.position[0], vertex.position[2]))
            .for_each(|(x, z)| {
                max_x = f32::max(max_x, x);
                min_x = f32::min(min_x, x);
                max_z = f32::max(max_z, z);
                min_z = f32::min(min_z, z);
            });

        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", model_name)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", model_name)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_elements = indices.len() as u32;
        Mesh {
            name: name.clone(),
            max_x,
            min_x,
            max_z,
            min_z,
            vertex_buffer,
            index_buffer,
            num_elements,
            material,
        }
    }

    fn get_extremes(&self) -> (f32, f32, f32, f32) {
        (self.max_x, self.min_x, self.max_z, self.min_z)
    }
}

#[derive(Debug)]
pub struct AnimatedMesh {
    pub name: String,
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32,
    pub materials: Vec<usize>,
    pub selected_material: usize
}

impl AnimatedMesh {
    pub fn new(name: String, model_name: &str, vertices: Vec<ModelVertex>, indices: Vec<u32>, materials: Vec<usize>, device: &wgpu::Device) -> AnimatedMesh {
        let mut max_x = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        vertices
            .iter()
            .map(|vertex| (vertex.position[0], vertex.position[2]))
            .for_each(|(x, z)| {
                max_x = f32::max(max_x, x);
                min_x = f32::min(min_x, x);
                max_z = f32::max(max_z, z);
                min_z = f32::min(min_z, z);
            });

        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", model_name)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", model_name)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_elements = indices.len() as u32;
        AnimatedMesh {
            name: name.clone(),
            max_x,
            min_x,
            max_z,
            min_z,
            vertex_buffer,
            index_buffer,
            num_elements,
            materials,
            selected_material: 0
        }
    }

    fn get_extremes(&self) -> (f32, f32, f32, f32) {
        (self.max_x, self.min_x, self.max_z, self.min_z)
    }

    fn animate(&mut self, material_index: usize) {
        self.selected_material = material_index;
    }
}

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub transparent_meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

impl Model {
    pub fn get_extremes(&self) -> (f32, f32, f32, f32) {
        let mut max_x = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        self.meshes.iter().map(|mesh| {
            mesh.get_extremes()
        }).for_each(|(max, mix, maz, miz)| {
            max_x = f32::max(max_x, max);
            min_x = f32::min(min_x, mix);
            max_z = f32::max(max_z, maz);
            min_z = f32::min(min_z, miz);
        });

        self.transparent_meshes.iter().map(|mesh| {
            mesh.get_extremes()
        }).for_each(|(max, mix, maz, miz)| {
            max_x = f32::max(max_x, max);
            min_x = f32::min(min_x, mix);
            max_z = f32::max(max_z, maz);
            min_z = f32::min(min_z, miz);
        });

        (max_x, min_x, max_z, min_z)
    }

    pub fn new_simple(vertices: Vec<ModelVertex>, indices: Vec<u32>, img: &ImageBuffer<Rgba<u8>, Vec<u8>>, gpu: &GpuState, layout: &wgpu::BindGroupLayout) -> Model {
        let mesh = Mesh::new("mesh1".to_string(), "unnamed", vertices, indices, 0, &gpu.device);
        let texture = Texture::from_dyn_image(&gpu.device, &gpu.queue, &img, None).unwrap();
        let material = Material::new(&gpu.device, "mat1", texture, layout);

        Model {
            meshes: vec![mesh],
            transparent_meshes: vec![],
            materials: vec![material],
        }
    }
}

#[derive(Debug)]
pub struct AnimatedModel {
    pub meshes: Vec<AnimatedMesh>,
    pub transparent_meshes: Vec<AnimatedMesh>,
    pub materials: Vec<Material>,
    pub instance: instances::Instance3D,
    pub instance_buffer: wgpu::Buffer,
    pub unculled_instance: bool
}

impl AnimatedModel {
    pub fn new(meshes: Vec<AnimatedMesh>, transparent_meshes: Vec<AnimatedMesh>, materials: Vec<Material>, device: &wgpu::Device) -> AnimatedModel {
        let mut instance = instances::Instance3D {
            position: cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 },
            animation: None,
            hidden: false
        };
        use wgpu::util::DeviceExt;
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&[instance.to_raw()]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        AnimatedModel {
            meshes,
            transparent_meshes,
            materials,
            instance,
            instance_buffer,
            unculled_instance: true,
        }
    }

    pub fn get_extremes(&self) -> (f32, f32, f32, f32) {
        let mut max_x = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        self.meshes.iter().map(|mesh| {
            mesh.get_extremes()
        }).for_each(|(max, mix, maz, miz)| {
            max_x = f32::max(max_x, max);
            min_x = f32::min(min_x, mix);
            max_z = f32::max(max_z, maz);
            min_z = f32::min(min_z, miz);
        });

        self.transparent_meshes.iter().map(|mesh| {
            mesh.get_extremes()
        }).for_each(|(max, mix, maz, miz)| {
            max_x = f32::max(max_x, max);
            min_x = f32::min(min_x, mix);
            max_z = f32::max(max_z, maz);
            min_z = f32::min(min_z, miz);
        });

        (max_x, min_x, max_z, min_z)
    }

    pub fn animate(&mut self, mesh_index: usize, material_index: usize) {
        self.meshes.get_mut(mesh_index).unwrap().animate(material_index);
    }

    pub(crate) fn uncull_instance(&mut self) {
        self.unculled_instance = true;
    }

    pub(crate) fn cull_all(&mut self) {
        self.unculled_instance = false;
    }

    pub(crate) fn hide(&mut self) {
        self.instance.hide()
    }

    pub(crate) fn show(&mut self) {
        self.instance.show()
    }

    pub(crate) fn is_hidden(&self) -> bool {
        self.instance.is_hidden()
    }
}

use crate::instances::{InstancedDraw, InstanceRaw};
use crate::instances::Instance;
impl InstancedDraw for AnimatedModel {
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

pub trait DrawText<'a> {
    fn draw_text(
        &mut self,
        material: &'a Material,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    );
}

impl<'a, 'b> DrawText<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_text(
        &mut self,
        material: &'a Material,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    ) {
        self.set_vertex_buffer(0, vertex_buffer.slice(..));
        self.set_bind_group(0, projection_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw(0..6, 0..1);
    }
}
