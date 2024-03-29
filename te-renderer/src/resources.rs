use cgmath::{Vector2, vec2};
use gltf;
use gltf::buffer;
use gltf::mesh::Mode as PrimitiveType;
use gltf::{Accessor, Semantic, Texture};

use crate::error::{TError, GLBErr};
use crate::{model, texture};

pub fn load_texture(
    image: &gltf::image::Data,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    file_name: &str,
) -> std::io::Result<texture::Texture> {
    texture::Texture::from_image(device, queue, image, Some(file_name))
}

/// Doesn't implement the full glb spec, so it will throw errors on some valid glb files
pub fn load_glb_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    resources_path: String,
    default_texture_path: &str,
) -> std::result::Result<model::Model, Box<dyn std::error::Error>> {
    let (document, buffers, images) = match gltf::import(resources_path + "/" + file_name) {
        Ok(result) => Ok(result),
        Err(err) => {
            eprintln!("Failed to load {}", file_name);
            Err(err)
        }
    }?; // TODO: don't hardcode the path

    let mut meshes = Vec::new();
    let mut transparent_meshes = Vec::new();
    let mut materials = Vec::new();
    let mut mat_index = 0;
    for glb_mesh in document.meshes() {
        let name = glb_mesh.name().ok_or(TError::NamelessGLB)?.to_string();
        for glb_primitive in glb_mesh.primitives() {
            if glb_primitive.mode() != PrimitiveType::Triangles {
                dbg!("Primitive must be triangles");
                return Err(Box::new(TError::InvalidGLB(GLBErr::UnsupportedPrimitiveType)))
            }

            // TODO: sort out the opaque, transparent and translucent primitives so we can correctly draw them using these steps:
            // 1. Draw all opaque primitives
            // 2. Sort the transparent primitives (from furthest to nearest)
            // 3. Draw the transparent primitives in order
            // to separate them see "alphaMode" in material
            let alpha = glb_primitive.material().alpha_mode();
            let pbr = glb_primitive.material().pbr_metallic_roughness();
            let glb_color = pbr.base_color_factor();
            let diffuse_texture = match pbr.base_color_texture() {
                Some(tex) => {
                    let glb_texture = tex.texture();
                    let image_index = glb_texture.source().index();
                    let texture_name = get_texture_name(glb_texture)?;

                    match glb_color {
                        [r, g, b, a] if r >= 0.999 && g >= 0.999 && b >= 0.999 && a >= 0.999 => {
                            // base color multiplier is so close to being 1 that it's not worth to process it
                            let image = images.get(image_index).ok_or(TError::InvalidGLB(GLBErr::TODO))?;
                            load_texture(image, device, queue, texture_name)?
                        }
                        [_, _, _, _] => {
                            let image = images.get(image_index).ok_or(TError::InvalidGLB(GLBErr::TODO))?;
                            let image = apply_base_color(image, glb_color);
                            load_texture(&image, device, queue, texture_name)?
                        }
                    }
                }
                None => {
                    let texture_name = "default_texture.png";
                    let image = image::io::Reader::open(default_texture_path)?
                        .decode()?
                        .into_rgba8();
                    let width = image.width();
                    let height = image.height();
                    let pixels = image.into_raw();
                    let image = gltf::image::Data {
                        pixels,
                        format: gltf::image::Format::R8G8B8A8, // TODO: change this placeholder when we actually take into account the format
                        width,
                        height,
                    };
                    match glb_color {
                        [r, g, b, a] if r >= 0.999 && g >= 0.999 && b >= 0.999 && a >= 0.999 => {
                            // base color multiplier is so close to being 1 that it's not worth to process it
                            load_texture(&image, device, queue, texture_name)?
                        }
                        [_, _, _, _] => {
                            let image = apply_base_color(&image, glb_color);
                            load_texture(&image, device, queue, texture_name)?
                        }
                    }
                }
            };
            let material = mat_index;
            mat_index += 1;

            materials.push(model::Material::new(device, &name, diffuse_texture, layout));

            // get the indices of the triangles
            let indices_accessor = glb_primitive.indices().ok_or(TError::InvalidGLB(GLBErr::TODO))?;
            let indices_buffer_index = get_buffer_index(&indices_accessor);
            let indices = load_scalar(indices_accessor, &buffers[indices_buffer_index])?; // TODO: handle the case of no indices

            // get the rest of mesh info
            let mut positions = None;
            let mut tex_coords = None;
            for attribute in glb_primitive.attributes() {
                let accessor = attribute.1;
                match attribute.0 {
                    Semantic::Positions => {
                        let index = get_buffer_index(&accessor);
                        positions = Some(load_vec3(accessor, &buffers[index])?);
                    }
                    Semantic::Normals => (), // ignoring since there is no light
                    Semantic::Tangents => (), // ignoring since there is no light
                    Semantic::Colors(_) => (), //println!("this model had colors and you ignored them!"), // TODO: ignore colors, they should be overwritten by textures
                    Semantic::TexCoords(_) => {
                        let index = get_buffer_index(&accessor);
                        tex_coords = Some(load_tex_coords(accessor, &buffers[index])?);
                    } //TODO: use the TexCoords parameter
                    Semantic::Joints(_) => (), //println!("this model had joints and you ignored them!"), // TODO: ignore animations for now
                    Semantic::Weights(_) => (), //println!("this model had weights and you ignored them!"), // TODO: ignore animations for now
                }
            }
            let tex_coords = match tex_coords {
                Some(coords) => coords,
                None => positions
                    .as_ref()
                    .ok_or(TError::InvalidGLB(GLBErr::TODO))?
                    .iter()
                    .map(|[x, y, _z]| [*x, *y])
                    .collect::<Vec<[f32; 2]>>(), // it doesn't matter what tex_coords are since the model doesn't have a texture anyway
            };
            let vertices = positions
                .ok_or(TError::InvalidGLB(GLBErr::TODO))?
                .into_iter()
                .zip(tex_coords)
                .map(|(position, tex_coords)| model::ModelVertex {
                    position,
                    tex_coords,
                })
                .collect::<Vec<model::ModelVertex>>();

            let mesh =
                model::Mesh::new(name.clone(), file_name, vertices, indices, material, device);

            match alpha {
                gltf::material::AlphaMode::Opaque => meshes.push(mesh),
                gltf::material::AlphaMode::Mask => meshes.push(mesh),
                gltf::material::AlphaMode::Blend => transparent_meshes.push(mesh),
            }
        }
    }
    Ok(model::Model {
        meshes,
        transparent_meshes,
        materials,
    })
}

fn load_vec3(accessor: Accessor, buffer: &buffer::Data) -> Result<Vec<[f32; 3]>, TError> {
    match accessor.data_type() {
        gltf::accessor::DataType::F32 => {}
        _ => {
            //panic!("vertex data should be F32!")
            return Err(TError::InvalidGLB(GLBErr::TODO))
        },
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Vec3 => {}
        _ => {
            //panic!("vertex data should be Vec3!")
            return Err(TError::InvalidGLB(GLBErr::TODO))
        },
    }
    let offset = accessor.offset() + accessor.view().expect("Sparse accessor").offset();
    let readable_size = accessor.count() * accessor.size();
    let vec3_data = &buffer[offset..offset + readable_size];
    let mut vec3 = Vec::new();
    for v in vec3_data.chunks_exact(12) {
        let x = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
        let y = f32::from_le_bytes([v[4], v[5], v[6], v[7]]);
        let z = f32::from_le_bytes([v[8], v[9], v[10], v[11]]);
        vec3.push([x, y, z]);
    }

    Ok(vec3)
}

fn load_tex_coords(accessor: Accessor, buffer: &buffer::Data) -> Result<Vec<[f32; 2]>, TError> {
    match accessor.data_type() {
        gltf::accessor::DataType::F32 => {}
        gltf::accessor::DataType::U8 => {
            return Err(TError::InvalidGLB(GLBErr::UnsupportedTexCoordDataType));
        },
        gltf::accessor::DataType::U16 => {
            return Err(TError::InvalidGLB(GLBErr::UnsupportedTexCoordDataType));
        },
        _ => {
            return Err(TError::InvalidGLB(GLBErr::InvalidTexCoordDataType));
        },
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Vec2 => {}
        _ => {
            return Err(TError::InvalidGLB(GLBErr::InvalidTexCoordAccessorDimension))
        },
    }
    let offset = accessor.offset() + accessor.view().expect("Sparse accessor").offset();
    let readable_size = accessor.count() * accessor.size();
    let vec2_data = &buffer[offset..offset + readable_size];
    let mut vec2 = Vec::new();
    for c in vec2_data.chunks_exact(8) {
        let x = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let y = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        vec2.push([x, y]);
    }

    Ok(vec2)
}

fn load_scalar(accessor: Accessor, buffer: &buffer::Data) -> Result<Vec<u32>, TError> {
    match accessor.data_type() {
        gltf::accessor::DataType::U16 => {}
        _ => {
            //println!("{:?}", accessor.data_type());
            //panic!("scalars should be U16!")
            return Err(TError::InvalidGLB(GLBErr::TODO))
        }
    }
    match accessor.dimensions() {
        gltf::accessor::Dimensions::Scalar => {}
        _ => {
            //panic!("scalars should be Scalar!")
            return Err(TError::InvalidGLB(GLBErr::TODO))
        },
    }
    let offset = accessor.offset() + accessor.view().expect("Sparse accessor").offset();
    let readable_size = accessor.count() * accessor.size();
    let scalar_data = &buffer[offset..offset + readable_size];
    let mut scalar = Vec::new();
    for v in scalar_data.chunks_exact(2) {
        let x = u16::from_le_bytes([v[0], v[1]]) as u32;
        scalar.push(x);
    }

    Ok(scalar)
}

fn get_buffer_index(accessor: &Accessor) -> usize {
    accessor.view().expect("Sparse accessor").buffer().index()
}

fn get_texture_name(tex: Texture) -> Result<&str, TError> {
    match tex.source().source() {
        gltf::image::Source::View { .. } => {
            //panic!("texture should be in a png")
            return Err(TError::InvalidGLB(GLBErr::TODO))
        },
        gltf::image::Source::Uri { uri, .. } => Ok(uri),
    }
}

fn apply_base_color(image: &gltf::image::Data, color: [f32; 4]) -> gltf::image::Data {
    assert!(image.format == gltf::image::Format::R8G8B8A8);
    let new_pixels = image
        .pixels
        .chunks_exact(4)
        .into_iter()
        .map(|c| {
            [
                ((f32::from(c[0]) * color[0]).floor() as u8),
                ((f32::from(c[1]) * color[1]).floor() as u8),
                ((f32::from(c[2]) * color[2]).floor() as u8),
                ((f32::from(c[3]) * color[3]).floor() as u8),
            ]
        })
        .flatten()
        .collect();

    gltf::image::Data {
        pixels: new_pixels,
        format: image.format,
        width: image.width,
        height: image.height,
    }
}

pub fn load_sprite(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    resources_path: String,
) -> std::result::Result<(model::Material, Vector2<f32>), Box<dyn std::error::Error>> {
    let full_path = resources_path + "/" + file_name;
    let img = image::open(&full_path)?;
    let img = img.as_rgba8().expect(&format!(
        "The image {full_path} doesn't contain an alpha channel. Only RGBA images are supported"
    ));
    let diffuse_texture = texture::Texture::from_dyn_image(device, queue, &img, Some(file_name));
    Ok((
        model::Material::new(device, file_name, diffuse_texture, layout),
        vec2(
            img.width() as f32,
            img.height() as f32
        )
    ))
}
