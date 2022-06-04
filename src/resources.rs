use gltf::buffer;
use gltf::{Semantic, Accessor, Texture};
use gltf;

use wgpu::util::DeviceExt;

use crate::{model, texture};

pub fn load_texture(
    image: &gltf::image::Data,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    file_name: &str,
) -> anyhow::Result<texture::Texture> {
    texture::Texture::from_image(device, queue, image, Some(file_name))
}

pub fn load_glb_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let (document, buffers, images) = gltf::import("ignore/resources/".to_string()+file_name).unwrap();

    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut mat_index = 0;
    for glb_mesh in document.meshes() {
        let name = glb_mesh.name().unwrap().to_string();
        for glb_primitive in glb_mesh.primitives() {
            // TODO: sort out the opaque, transparent and translucent primitives so we can correctly draw them using these steps:
            // 1. Draw all opaque primitives
            // 2. Sort the transparent primitives (from furthest to nearest)
            // 3. Draw the transparent primitives in order
            // to separate them see "alphaMode" in material
            let pbr = glb_primitive.material().pbr_metallic_roughness();
            let glb_color = pbr.base_color_factor();
            let glb_texture = pbr.base_color_texture().unwrap().texture();
            let material = mat_index;
            mat_index += 1;
            let image_index = glb_texture.source().index();
            let texture_name = get_texture_name(glb_texture);
            
            let diffuse_texture = match glb_color {
                [r, g, b, a] if r >= 0.999 && g >= 0.999 && b >= 0.999 && a >= 0.999 => {
                    // base color multiplier is so close to being 1 that it's not worth to process it
                    let image = images.get(image_index).unwrap();
                    load_texture(image, device, queue, texture_name)?
                },
                [_,_,_,_] => {
                    let image = images.get(image_index).unwrap();
                    let image = apply_base_color(image, glb_color);
                    load_texture(&image, device, queue, texture_name)?
                }
            };

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

fn load_vec3(accessor: Accessor, buffer: &buffer::Data) -> Vec<[f32; 3]> {
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

fn load_vec2(accessor: Accessor, buffer: &buffer::Data) -> Vec<[f32; 2]> {
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

fn load_scalar(accessor: Accessor, buffer: &buffer::Data) -> Vec<u32> {
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

fn apply_base_color(image: &gltf::image::Data, color: [f32; 4]) -> gltf::image::Data {
    assert!(image.format == gltf::image::Format::R8G8B8A8);
    let new_pixels = image.pixels.chunks_exact(4).into_iter()
        .map(|c| {
            [
             ((f32::from(c[0]) * color[0]).floor() as u8),
             ((f32::from(c[1]) * color[1]).floor() as u8),
             ((f32::from(c[2]) * color[2]).floor() as u8),
             ((f32::from(c[3]) * color[3]).floor() as u8)
            ]
        })
        .flatten()
        .collect();
    
    gltf::image::Data {
        pixels: new_pixels,
        format: image.format,
        width: image.width,
        height: image.height
    }
}
