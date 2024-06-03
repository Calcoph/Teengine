use std::collections::HashMap;

use cgmath::Vector2;
use glyph_brush::{ab_glyph::FontArc, Section};
use image::{ImageBuffer, Rgba};
use wgpu::util::StagingBelt;
use wgpu_glyph::{GlyphBrush, GlyphBrushBuilder};

use crate::{model::Material, state::GpuState, texture};

pub trait Font {
    type Character;

    fn write_to_material(
        &self,
        characters: Vec<Self::Character>,
        gpu: &GpuState,
        layout: &wgpu::BindGroupLayout,
    ) -> (Material, f32, f32);

    fn load_font(font_path: String) -> Self;
}

#[derive(Debug)]
pub struct OldTextFont {
    characters: HashMap<String, image::ImageBuffer<Rgba<u8>, Vec<u8>>>,
}

impl Font for OldTextFont {
    type Character = String;

    fn write_to_material(
        &self,
        characters: Vec<String>,
        gpu: &GpuState,
        layout: &wgpu::BindGroupLayout,
    ) -> (Material, f32, f32) {
        let mut width = 0;
        let mut height = 0;
        let mut container = Vec::new();
        for charac in characters.iter() {
            let bitmap = self
                .characters
                .get(charac)
                .expect("That character is not available in the font")
                .clone();
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
            let bitmap = self
                .characters
                .get(charac)
                .expect("That character is not available in the font")
                .clone();
            let height_diff = height - bitmap.height();
            for i in 0..height_diff {
                for _ in 0..bitmap.width() {
                    v.get_mut(i as usize)
                        .expect("Unreachable")
                        .push(Rgba([0, 0, 0, 0]));
                }
            }
            for (i, row) in bitmap.enumerate_rows() {
                for (_i, _j, pixel) in row {
                    v.get_mut((i + height_diff) as usize)
                        .expect("Unreachable")
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

        let new_image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width, height, container)
            .expect("Unreachable");

        let tex = texture::Texture::from_dyn_image(&gpu.device, &gpu.queue, &new_image, None);
        (
            Material::new(&gpu.device, "text", tex, layout),
            width as f32,
            height as f32,
        )
    }

    fn load_font(font_dir_path: String) -> OldTextFont {
        let mut characters = HashMap::new();
        for file in std::fs::read_dir(font_dir_path.clone()).expect("Could not read font directory")
        {
            let file_name = file.expect("Could not read font directory").file_name();
            let file_name = file_name.to_str().expect("Invalid file name");
            if std::path::Path::new(&(font_dir_path.clone() + "/" + file_name)).is_file()
                && file_name.ends_with(".png")
            {
                let img = image::io::Reader::open(font_dir_path.clone() + "/" + file_name)
                    .expect("Unable to read file")
                    .decode()
                    .expect("Unsupported image format")
                    .into_rgba8();
                characters.insert(String::from(file_name.trim_end_matches(".png")), img);
            }
        }

        OldTextFont { characters }
    }
}

#[derive(Clone)]
pub struct FontReference {
    index: usize,
}

#[derive(Debug)]
pub struct FontError;

#[derive(Debug)]
pub struct TextState {
    buffer: StagingBelt,
    brushes: Vec<GlyphBrush<()>>,
}

impl TextState {
    pub fn new() -> TextState {
        TextState {
            buffer: StagingBelt::new(1024),
            brushes: Vec::new(),
        }
    }

    pub(crate) fn load_font(
        &mut self,
        font_path: String,
        device: &wgpu::Device,
        render_format: wgpu::TextureFormat,
    ) -> Result<FontReference, FontError> {
        let font_data = match std::fs::read(&font_path) {
            Ok(f) => Ok(f),
            Err(_) => Err(FontError),
        }?;
        let font = match FontArc::try_from_vec(font_data) {
            Ok(f) => Ok(f),
            Err(_) => Err(FontError),
        }?;
        let brush = GlyphBrushBuilder::using_font(font).build(device, render_format);
        let reference = FontReference {
            index: self.brushes.len(),
        };
        self.brushes.push(brush);
        Ok(reference)
    }

    pub(crate) fn end_render(&mut self) {
        self.buffer.finish()
    }

    pub fn after_present(&mut self) {
        self.buffer.recall()
    }

    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        window_size: Vector2<u32>,
        sections: &[(FontReference, Vec<Section>)],
    ) {
        for (font, texts) in sections {
            let brush = self
                .brushes
                .get_mut(font.index)
                .expect("Invalid font reference");
            for text in texts {
                brush.queue(text)
            }
            brush
                .draw_queued(
                    device,
                    &mut self.buffer,
                    encoder,
                    view,
                    window_size.x,
                    window_size.y,
                )
                .expect("Error drawing old text");
        }
    }
}
