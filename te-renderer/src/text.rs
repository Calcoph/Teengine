use std::collections::HashMap;

use image::{Rgba, ImageBuffer};

use crate::{state::GpuState, model::Material, texture};

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
