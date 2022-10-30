use std::{collections::HashMap, cmp::Reverse};

use glyph_brush::{ab_glyph::FontArc, Section};
use image::{Rgba, ImageBuffer};
use sorted_vec::ReverseSortedVec;
use wgpu::util::StagingBelt;
use wgpu_glyph::{GlyphBrushBuilder, GlyphBrush};

use crate::{state::{GpuState, TeColor}, model::Material, texture};

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

#[derive(Clone)]
pub struct FontReference {
    path: String
}

pub struct TextReference {
    font: FontReference,
    index: usize
}

#[derive(Debug)]
pub struct FontError;

#[derive(Debug)]
pub struct TextState {
    buffer: StagingBelt,
    brushes: HashMap<String, GlyphBrush<()>>,
    texts: HashMap<String, Vec<Option<TeText>>>,
    deleted_texts: HashMap<String, ReverseSortedVec<usize>>
}

impl TextState {
    pub fn new() -> TextState {
        TextState {
            buffer: StagingBelt::new(1024),
            brushes: HashMap::new(),
            texts: HashMap::new(),
            deleted_texts: HashMap::new()
        }
    }

    pub(crate) fn load_font(&mut self, font_path: String, device: &wgpu::Device, render_format: wgpu::TextureFormat) -> Result<FontReference, FontError>{
        let font_data = match std::fs::read(&font_path) {
            Ok(f) => Ok(f),
            Err(_) => Err(FontError),
        }?;
        let font = match FontArc::try_from_vec(font_data) {
            Ok(f) => Ok(f),
            Err(_) => Err(FontError),
        }?;
        let brush = GlyphBrushBuilder::using_font(font)
            .build(&device, render_format);
        let reference = FontReference { path: font_path.clone() };
        self.brushes.insert(font_path.clone(), brush);
        self.texts.insert(font_path.clone(), Vec::new());
        self.deleted_texts.insert(font_path, ReverseSortedVec::new());
        Ok(reference)
    }

    pub(crate) fn end_render(&mut self) {
        self.buffer.finish()
    }

    pub fn after_present(&mut self) {
        self.buffer.recall()
    }

    pub(crate) fn draw(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, window_size: (u32, u32)) {
        for (font_path, brush) in self.brushes.iter_mut() {
            if let Some(texts) = self.texts.get(font_path) {
                for text in texts {
                    if let Some(text) = text {
                        if !text.hidden {
                            brush.queue(Section {
                                screen_position: text.position.into(),
                                bounds: (window_size.0 as f32, window_size.1 as f32),
                                text: vec![glyph_brush::Text::new(&text.string)
                                    .with_color([text.color.get_red() as f32, text.color.get_green() as f32, text.color.get_blue() as f32, 1.0])
                                    .with_scale(text.size)
                                ],
                                ..Section::default()
                            })
                        }
                    }
                }

                brush.draw_queued(
                    device,
                    &mut self.buffer,
                    encoder,
                    view,
                    window_size.0,
                    window_size.1
                ).unwrap();
            }
        }
    }

    pub(crate) fn place_text<P: Into<cgmath::Vector2<f32>>>(&mut self, font: &FontReference, message: String, position: P, color: TeColor, size: f32) -> TextReference {
        let new_text = TeText {
            position: position.into(),
            string: message,
            color,
            size,
            hidden: false,
        };
        let texts = self.texts.get_mut(&font.path).unwrap();
        let deleted = self.deleted_texts.get_mut(&font.path).unwrap();
        match deleted.pop() {
            Some(index) => {
                texts[index.0] = Some(new_text);
                TextReference { font: font.to_owned(), index: index.0 }
            },
            None => {
                texts.push(Some(new_text));
                TextReference { font: font.to_owned(), index: texts.len()-1 }
            },
        }
    }

    pub(crate) fn forget_text(&mut self, text: TextReference) {
        self.deleted_texts.get_mut(&text.font.path).unwrap().insert(Reverse(text.index));
        self.texts.get_mut(&text.font.path).unwrap()[text.index] = None;
    }

    pub(crate) fn move_text(&mut self, text: &TextReference, direction: cgmath::Vector2<f32>) {
        let text = self.texts.get_mut(&text.font.path).unwrap().get_mut(text.index).unwrap().as_mut().unwrap();
        text.position += direction;
    }

    pub(crate) fn set_text_position(&mut self, text: &TextReference, position: cgmath::Vector2<f32>) {
        let text = self.texts.get_mut(&text.font.path).unwrap().get_mut(text.index).unwrap().as_mut().unwrap();
        text.position = position;
    }

    pub(crate) fn get_text_position(&self, text: &TextReference) -> cgmath::Vector2<f32> {
        let text = self.texts.get(&text.font.path).unwrap().get(text.index).unwrap().as_ref().unwrap();
        text.position
    }

    pub(crate) fn resize_text(&mut self, text: &TextReference, size: f32) {
        let text = self.texts.get_mut(&text.font.path).unwrap().get_mut(text.index).unwrap().as_mut().unwrap();
        text.size = size;
    }

    pub(crate) fn get_text_size(&self, text: &TextReference) -> f32 {
        let text = self.texts.get(&text.font.path).unwrap().get(text.index).unwrap().as_ref().unwrap();
        text.size
    }

    pub(crate) fn set_text_animation() {
        todo!()
    }

    pub(crate) fn hide_text(&mut self, text: &TextReference) {
        let text = self.texts.get_mut(&text.font.path).unwrap().get_mut(text.index).unwrap().as_mut().unwrap();
        text.hidden = true;
    }

    pub(crate) fn show_text(&mut self, text: &TextReference) {
        let text = self.texts.get_mut(&text.font.path).unwrap().get_mut(text.index).unwrap().as_mut().unwrap();
        text.hidden = false;
    }

    pub(crate) fn is_text_hidden(&self, text: &TextReference) -> bool{
        let text = self.texts.get(&text.font.path).unwrap().get(text.index).unwrap().as_ref().unwrap();
        text.hidden
    }
}

#[derive(Debug)]
struct TeText {
    position: cgmath::Vector2<f32>,
    string: String,
    color: TeColor,
    size: f32,
    hidden: bool
}
