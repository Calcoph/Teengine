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
    is_normal_map: bool,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

/* pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        }
    ).await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        let diffuse_texture = load_texture(&m.diffuse_texture, device, queue, false).await?;
        let normal_texture = load_texture(&m.normal_texture, device, queue, true).await?;

        materials.push(model::Material::new(
            device,
            &m.name,
            diffuse_texture,
            normal_texture,
            layout
        ));
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [
                        m.mesh.texcoords[i * 2],
                        m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2]
                    ],
                    // We'll calculate these later
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3]
                }).collect::<Vec<_>>();

            let indices = &m.mesh.indices;
            let mut triangles_included = (0..vertices.len()).collect::<Vec<_>>();

            // Calculate tangents and bitangents. We're going to
            // use the triangles, so we need to loop through
            // the indices in chunks of 3
            for c in indices.chunks_exact(3) {
                let v0 = vertices[c[0] as usize];
                let v1 = vertices[c[1] as usize];
                let v2 = vertices[c[2] as usize];

                let pos0: cgmath::Vector3<_> = v0.position.into();
                let pos1: cgmath::Vector3<_> = v1.position.into();
                let pos2: cgmath::Vector3<_> = v2.position.into();

                let uv0: cgmath::Vector2<_> = v0.tex_coords.into();
                let uv1: cgmath::Vector2<_> = v1.tex_coords.into();
                let uv2: cgmath::Vector2<_> = v2.tex_coords.into();                

                // Calculate the edges of the triangle
                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;

                // This will give us a direction to calculate the
                // tangent and bitangent
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                // Solving the following system of equations will
                // give us the tangent and bitangent.
                //      delta_pos1 = delta_uv1.x * T + delta_uv1.y * B
                //      delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                // Luckily, the place I found this equation provided
                // the solution!
                let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                // We flip the bitangent to enable right-handed normal
                // maps with wgpu texture coordinate system
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                vertices[c[0] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[0] as usize].tangent)).into();
                vertices[c[1] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[1] as usize].tangent)).into();
                vertices[c[2] as usize].tangent =
                    (tangent + cgmath::Vector3::from(vertices[c[2] as usize].tangent)).into();
                vertices[c[0] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[0] as usize].bitangent)).into();
                vertices[c[1] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[1] as usize].bitangent)).into();
                vertices[c[2] as usize].bitangent =
                    (bitangent + cgmath::Vector3::from(vertices[c[2] as usize].bitangent)).into();

                // Used to average the tangents/bitangents
                triangles_included[c[0] as usize] += 1;
                triangles_included[c[1] as usize] += 1;
                triangles_included[c[2] as usize] += 1;
            }

            // Average the tangents/bitangents
            for (i, n) in triangles_included.into_iter().enumerate() {
                let denom = 1.0 / n as f32;
                let mut v = &mut vertices[i];
                v.tangent = (cgmath::Vector3::from(v.tangent) * denom)
                    .normalize()
                    .into();
                v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom)
                    .normalize()
                    .into();
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX
            });

            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0)
            }
        }).collect::<Vec<_>>();
    
    Ok(model::Model { meshes, materials })
} */

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
    for glb_mesh in document.meshes() {
        let name = glb_mesh.name().unwrap().to_string();
        for glb_primitive in glb_mesh.primitives() {
            //let _glb_mode = glb_primitive.mode();
            let glb_texture = glb_primitive.material().pbr_metallic_roughness().base_color_texture().unwrap().texture();
            let material = glb_texture.index();
            let texture_name = get_texture_name(glb_texture);

            //let diffuse_name = get_texture_name(mat.occlusion_texture().unwrap().texture());
            let diffuse_texture = load_texture(texture_name, device, queue, false).await?;
            //let normal_texture = load_texture(normal_name, device, queue, true).await?;

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