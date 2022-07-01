use std::{collections::HashMap, ops::Range};

use image::{buffer::Pixels, ImageBuffer, Rgba};

use crate::{state::GpuState, texture};

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

pub trait DrawSprite<'a> {
    fn draw_sprite_instanced(
        &mut self,
        material: &'a Material,
        instances: Range<u32>,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    );
}

impl<'a, 'b> DrawSprite<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_sprite_instanced(
        &mut self,
        material: &'a Material,
        instances: Range<u32>,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    ) {
        self.set_vertex_buffer(0, vertex_buffer.slice(..));
        self.set_bind_group(0, projection_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw(0..6, instances);
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

#[derive(Debug)]
pub struct Model {
    pub meshes: Vec<Mesh>,
    pub transparent_meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub trait DrawModel<'a> {
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_model_instanced_with_material(
        &mut self,
        model: &'a Model,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group);
        }
    }

    fn draw_model_instanced_with_material(
        &mut self,
        model: &'b Model,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group)
        }
    }

    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }
}

pub trait DrawTransparentModel<'a> {
    fn tdraw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_model_instanced_with_material(
        &mut self,
        model: &'a Model,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawTransparentModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn tdraw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.transparent_meshes {
            let material = &model.materials[mesh.material];
            self.tdraw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group);
        }
    }

    fn tdraw_model_instanced_with_material(
        &mut self,
        model: &'b Model,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            self.tdraw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group)
        }
    }

    fn tdraw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.tdraw_mesh_instanced(mesh, material, 0..1, camera_bind_group);
    }

    fn tdraw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
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

#[derive(Debug)]
pub struct Font {
    characters: HashMap<String, image::ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl Font {
    pub fn write_to_material(
        &self,
        characters: Vec<String>,
        gpu: &GpuState,
        layout: &wgpu::BindGroupLayout,
    ) -> (Material, f32, f32) {
        let mut width = 0;
        let mut height = 0;
        let mut container = Vec::new();
        for charac in characters.iter() {
            let bitmap = self.characters.get(charac).unwrap().clone();
            width += bitmap.width();
            if bitmap.height() > height {
                height = bitmap.height();
            };
        }

        let mut v = Vec::new();
        for _ in 0..height {
            v.push(Vec::new())
        }

        for charac in characters.iter() {
            let bitmap = self.characters.get(charac).unwrap().clone();
            let height_diff = height - bitmap.height();
            for i in 0..height_diff {
                for _ in 0..bitmap.width() {
                    v.get_mut(i as usize).unwrap().push(Rgba([0, 0, 0, 0]));
                }
            }
            for (i, row) in bitmap.enumerate_rows() {
                for (_i, _j, pixel) in row {
                    v.get_mut((i + height_diff) as usize)
                        .unwrap()
                        .push(pixel.clone())
                }
            }
        }

        for row in v {
            for pixel in row {
                container.push(pixel.0[0]);
                container.push(pixel.0[1]);
                container.push(pixel.0[2]);
                container.push(pixel.0[3]);
            }
        }

        let new_image =
            ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width, height, container).unwrap();

        let tex =
            texture::Texture::from_dyn_image(&gpu.device, &gpu.queue, &new_image, None).unwrap();
        (
            Material::new(&gpu.device, "text", tex, layout),
            width as f32,
            height as f32,
        )
    }

    pub fn new(font_dir_path: String) -> Font {
        let mut characters = HashMap::new();
        for file in std::fs::read_dir(font_dir_path.clone()).unwrap() {
            let file_name = file.unwrap().file_name();
            let file_name = file_name.to_str().unwrap();
            if std::path::Path::new(&(font_dir_path.clone() + "/" + file_name)).is_file()
                && file_name.ends_with(".png")
            {
                let img = image::io::Reader::open(font_dir_path.clone() + "/" + file_name)
                    .unwrap()
                    .decode()
                    .unwrap()
                    .into_rgba8();
                characters.insert(String::from(file_name.trim_end_matches(".png")), img);
            }
        }

        Font { characters }
    }
}
