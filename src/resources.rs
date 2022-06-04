use std::io::{BufReader, Cursor};
use std::{path, fs};
use cgmath::prelude::*;
use gltf::buffer::Data;
use gltf::{Semantic, Accessor, Texture};

use wgpu::util::DeviceExt;

use crate::model::ModelVertex;
use crate::{model, texture};

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    let path = path::Path::new(env!("OUT_DIR"))
        .join("resources")
        .join(file_name);
    let txt = fs::read_to_string(path)?;

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(env!("OUT_DIR"))
        .join("resources")
        .join(file_name);
    let data = std::fs::read(path)?;

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_glb_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let (document, buffers, _images) = gltf::import("resources/".to_string()+file_name).unwrap();
    // materials:
    //  pub name: String,
    //  pub diffuse_texture: texture::Texture,
    //  pub normal_texture: texture::Texture,
    //  pub bind_group: wgpu::BindGroup
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut mat_index = 0;
    for glb_mesh in document.meshes() {
        let name = glb_mesh.name().unwrap().to_string();
        for glb_primitive in glb_mesh.primitives() {
            //let _glb_mode = glb_primitive.mode();
            let glb_texture = glb_primitive.material().pbr_metallic_roughness().base_color_texture().unwrap().texture();
            let material = mat_index;
            mat_index += 1;
            let texture_name = get_texture_name(glb_texture);

            let diffuse_texture = load_texture(texture_name, device, queue).await?;

            materials.push(model::Material::new(
                device,
                &name,
                diffuse_texture,
                layout
            ));

            // get the indices of the triangles
            let indices_accessor = glb_primitive.indices().unwrap();
            let indices_buffer_index = get_buffer_index(&indices_accessor);
            let indices = load_scalar(indices_accessor, &buffers[indices_buffer_index]); // TODO: handle the case of no indices
            
            // get the rest of mesh info
            let mut positions = None;
            let mut tex_coords = None;
            for attribute in glb_primitive.attributes() {
                let accessor = attribute.1;
                match attribute.0 {
                    Semantic::Positions => {
                        let index = get_buffer_index(&accessor);
                        positions = Some(load_vec3(accessor, &buffers[index]));
                    },
                    Semantic::Normals => (), // ignoring since there is no light
                    Semantic::Tangents => (), // ignoring since there is no light
                    Semantic::Colors(_) => println!("this model had colors and you ignored them!"), // TODO: ignore colors, they should be overwritten by textures
                    Semantic::TexCoords(_) => {
                        let index = get_buffer_index(&accessor);
                        tex_coords = Some(load_vec2(accessor, &buffers[index]));
                    }, //TODO: use the TexCoords parameter
                    Semantic::Joints(_) => println!("this model had joints and you ignored them!"), // TODO: ignore animations for now
                    Semantic::Weights(_) => println!("this model had weights and you ignored them!"), // TODO: ignore animations for now
                }
            }
            let vertices = positions.unwrap()
                .into_iter()
                .zip(tex_coords.unwrap())
                .map(|(position, tex_coords)| {
                    model::ModelVertex {
                        position,
                        tex_coords
                    }
                }).collect::<Vec<model::ModelVertex>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX
            });
            
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX
            });

            let num_elements = indices.len() as u32;
            let mesh = model::Mesh {
                name: name.clone(),
                vertex_buffer,
                index_buffer,
                num_elements,
                material
            };
            meshes.push(mesh);
        }
    }
    Ok(model::Model {meshes, materials})
}

fn load_vec3(accessor: Accessor, buffer: &Data) -> Vec<[f32; 3]> {
    match accessor.data_type() {
        gltf::accessor::DataType::F32 => {},
        _ => panic!("vertex data should be F32!")
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Vec3 => {},
        _ => panic!("vertex data should be Vec3!")
    }
    let offset = accessor.offset()+accessor.view().unwrap().offset();
    let readable_size = accessor.count() * accessor.size();
    let vec3_data = &buffer[offset..offset+readable_size];
    let mut vec3 = Vec::new();
    for v in vec3_data.chunks_exact(12) {
        let x = f32::from_le_bytes([v[0],v[1],v[2],v[3]]);
        let y = f32::from_le_bytes([v[4],v[5],v[6],v[7]]);
        let z = f32::from_le_bytes([v[8],v[9],v[10],v[11]]);
        vec3.push([x, y, z]);
    }

    vec3
}

fn load_vec2(accessor: Accessor, buffer: &Data) -> Vec<[f32; 2]> {
    match accessor.data_type() {
        gltf::accessor::DataType::F32 => {},
        _ => panic!("tex coords should be F32!")
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Vec2 => {},
        _ => panic!("tex coords should be Vec2!")
    }
    let offset = accessor.offset()+accessor.view().unwrap().offset();
    let readable_size = accessor.count() * accessor.size();
    let vec2_data = &buffer[offset..offset+readable_size];
    let mut vec2 = Vec::new();
    for c in vec2_data.chunks_exact(8) {
        let x = f32::from_le_bytes([c[0],c[1],c[2],c[3]]);
        let y = f32::from_le_bytes([c[4],c[5],c[6],c[7]]);
        vec2.push([x, y]);
    }

    vec2
}

fn load_scalar(accessor: Accessor, buffer: &Data) -> Vec<u32> {
    match accessor.data_type() {
        gltf::accessor::DataType::U16 => {},
        _ => {
            println!("{:?}",accessor.data_type());
            panic!("scalars should be U16!")
        }
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Scalar => {},
        _ => panic!("scalars should be Scalar!")
    }
    let offset = accessor.offset()+accessor.view().unwrap().offset();
    let readable_size = accessor.count() * accessor.size();
    let scalar_data = &buffer[offset..offset+readable_size];
    let mut scalar = Vec::new();
    for v in scalar_data.chunks_exact(2) {
        let x = u16::from_le_bytes([v[0],v[1]]) as u32;
        scalar.push(x);
    }
    for index in &scalar {
        print!("{}, ", index)
    }

    scalar
}

fn get_buffer_index(accessor: &Accessor) -> usize {
    accessor.view().unwrap().buffer().index()
}

fn get_texture_name(tex: Texture) -> &str {
    match tex.source().source() {
        gltf::image::Source::View { .. } => panic!("texture should be in a png"),
        gltf::image::Source::Uri { uri, .. } => uri,
    }
}