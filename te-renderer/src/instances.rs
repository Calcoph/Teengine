use std::collections::{HashMap, HashSet};

use cgmath::Vector3;

pub mod animation;
pub mod model;
pub mod sprite;
pub mod text;

use crate::{
    resources::{load_glb_model, load_sprite},
    state::GpuState,
    temap, camera
};

use self::{animation::Animation, model::InstancedModel, sprite::{InstancedSprite, AnimatedSprite}, text::{InstancedText, TextReference}};

#[derive(Debug)]
struct RenderMatrix {
    cells: Vec<Vec<Vec<InstanceReference>>>,
    cols: usize,
    chunk_size: (f32, f32, f32)
}

impl RenderMatrix {
    fn new(chunk_size: (f32, f32, f32)) -> Self {
        RenderMatrix {
            cells: Vec::new(),
            cols: 0,
            chunk_size
        }
    }

    fn register_instance(&mut self, reference: InstanceReference, position: cgmath::Vector3<f32>, (max_x, min_x, max_z, min_z): (f32, f32, f32, f32)) {
        let max_x = max_x + position.x;
        let min_x = min_x + position.x;
        let max_z = max_z + position.z;
        let min_z = min_z + position.z;
        let corners = vec![
            (max_x, max_z),
            (max_x, min_z),
            (min_x, max_z),
            (min_x, min_z)
        ];
        let chunks = corners.into_iter().map(|(x, z)| {
            // TODO: take into account that f32 can be negative, but row and col can never be negative
            let row = ((z/self.chunk_size.2).floor()) as usize;
            let col = ((x/self.chunk_size.0).floor()) as usize;
            (row, col)
        }).collect::<HashSet<(usize, usize)>>();
        chunks.into_iter().for_each(|(row, col)| {
            while self.cells.len() <= row {
                self.cells.push(Vec::new())
            };
            let row_vec = self.cells.get_mut(row).unwrap();
            while row_vec.len() <= col {
                row_vec.push(Vec::new());
            };
            row_vec.get_mut(col).unwrap().push(reference.clone());
            if col+1 > self.cols {
                self.cols = col+1
            }
        });
    }

    fn unregister_instance(&mut self, reference: &InstanceReference, position: cgmath::Vector3<f32>, (max_x, min_x, max_z, min_z): (f32, f32, f32, f32)) {
        let max_x = max_x + position.x;
        let min_x = min_x + position.x;
        let max_z = max_z + position.z;
        let min_z = min_z + position.z;
        let corners = vec![
            (max_x, max_z),
            (max_x, min_z),
            (min_x, max_z),
            (min_x, min_z)
        ];
        let chunks = corners.into_iter().map(|(x, z)| {
            // TODO: take into account that f32 can be negative, but row and col can never be negative
            let row = ((z/self.chunk_size.2).floor()) as usize;
            let col = ((x/self.chunk_size.0).floor()) as usize;
            (row, col)
        }).collect::<HashSet<(usize, usize)>>();
        chunks.into_iter().for_each(|(row, col)| {
            let row_vec = self.cells.get_mut(row).unwrap();
            let chunk = row_vec.get_mut(col).unwrap();
            let pos = chunk.iter().enumerate().find(|(_, item)| **item == *reference).unwrap().0;
            chunk.remove(pos);
        });
    }

    fn update_rendered(&self, view_cone: &camera::Frustum) -> Vec<InstanceReference> {
        let mut v = Vec::new();
        for row in 0..self.cells.len() {
            for col in 0..self.cols {
                let cell = self.cells.get(row).unwrap().get(col);
                match cell {
                    Some(references )if references.len() > 0 => {
                        if view_cone.is_inside(row, col) {
                            for reference in references {
                                v.push(reference.clone());
                            }
                        }
                    },
                    _ => (),
                }
            }
        }

        v
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl crate::model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub trait Instance {
    fn to_raw(&mut self) -> InstanceRaw;

    fn move_direction<V: Into<cgmath::Vector3<f32>>>(&mut self, direction: V);

    fn move_to<P: Into<cgmath::Vector3<f32>>>(&mut self, position: P);
}

#[derive(Debug)]
pub struct Instance2D {
    pub position: cgmath::Vector2<f32>,
    pub size: cgmath::Vector2<f32>,
    animation: Option<Animation>
}

impl Instance2D {
    fn resize<V: Into<cgmath::Vector2<f32>>>(&mut self, new_size: V) {
        let size = new_size.into();
        self.size = size;
    }

    fn get_animated(&mut self) -> Instance2D {
        let animation = self.animation.as_mut().unwrap();
        let translation = animation.get_translation();
        let scale = animation.get_scale();
        let x = self.position.x + translation.x;
        let y = self.position.y + translation.y;
        let w = self.size.x + scale.x;
        let h = self.size.y + scale.y;
        let position = cgmath::Vector2{ x, y };
        let size = cgmath::Vector2 { x: w, y: h };
        Instance2D {
            position,
            size,
            animation: None
        }
    }
}

impl Instance for Instance2D {
    fn to_raw(&mut self) -> InstanceRaw {
        let animation;
        let instance = if self.animation.is_some() {
            animation = self.get_animated();
            &animation
        } else {
            self
        };
        let sprite = cgmath::Matrix4::from_translation(Vector3 {
            x: instance.position.x,
            y: instance.position.y,
            z: 0.0,
        }) * cgmath::Matrix4::from_nonuniform_scale(instance.size.x, instance.size.y, 1.0);

        InstanceRaw {
            model: sprite.into(),
        }
    }

    fn move_direction<V: Into<cgmath::Vector3<f32>>>(&mut self, direction: V) {
        let direction = direction.into();
        self.position = self.position + cgmath::Vector2::new(direction.x, direction.y);
    }

    fn move_to<P: Into<cgmath::Vector3<f32>>>(&mut self, position: P) {
        let position = position.into();
        self.position = cgmath::Vector2::new(position.x, position.y)
    }
}

#[derive(Debug)]
pub struct Instance3D {
    pub position: cgmath::Vector3<f32>,
    pub animation: Option<Animation>,
}

impl Instance3D {
    fn get_animated(&mut self) -> Instance3D {
        let animation = self.animation.as_mut().unwrap();
        let translation = animation.get_translation();
        let x = self.position.x + translation.x;
        let y = self.position.y + translation.y;
        let z = self.position.z + translation.z;
        let position = cgmath::Vector3{ x, y, z };
        Instance3D {
            position,
            animation: None,
        }
    }
}

impl Instance for Instance3D {
    fn to_raw(&mut self) -> InstanceRaw {
        let animation;
        let instance = if self.animation.is_some() {
            animation = self.get_animated();
            &animation
        } else {
            self
        };
        let model = cgmath::Matrix4::from_translation(instance.position);
        InstanceRaw {
            model: model.into(),
        }
    }

    fn move_direction<V: Into<cgmath::Vector3<f32>>>(&mut self, direction: V) {
        self.position = self.position + direction.into();
    }

    fn move_to<P: Into<cgmath::Vector3<f32>>>(&mut self, position: P) {
        self.position = position.into();
    }
}

trait InstancedDraw {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    );

    fn set_instance_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        position: P,
        queue: &wgpu::Queue,
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Dimension {
    D2,
    Anim2D,
    D3,
}

/// Handle of a 3D model or 2D sprite. You will need it when changing their properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceReference {
    name: String,
    index: usize,
    dimension: Dimension,
}

impl InstanceReference {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_id(&self) -> usize {
        self.index
    }
}

/// Manages the window's 3D models, 2D sprites and 2D texts
#[derive(Debug)]
pub struct InstancesState {
    pub instances: HashMap<String, InstancedModel>,
    pub instances_2d: HashMap<String, InstancedSprite>,
    pub animated_sprites: HashMap<String, AnimatedSprite>,
    pub texts: Vec<Option<InstancedText>>,
    deleted_texts: Vec<usize>,
    pub layout: wgpu::BindGroupLayout,
    tile_size: (f32, f32, f32),
    chunk_size: (f32, f32, f32),
    pub resources_path: String,
    pub default_texture_path: String,
    font: crate::model::Font,
    render_matrix: RenderMatrix
}

impl InstancesState {
    pub fn new(
        layout: wgpu::BindGroupLayout,
        tile_size: (f32, f32, f32),
        chunk_size: (f32, f32, f32),
        resources_path: String,
        default_texture_path: String,
        font_dir_path: String,
    ) -> Self {
        let instances = HashMap::new();
        let instances_2d = HashMap::new();
        let animated_sprites = HashMap::new();
        let texts = Vec::new();
        let deleted_texts = Vec::new();
        let font = crate::model::Font::new(font_dir_path);
        let render_matrix = RenderMatrix::new(chunk_size);

        InstancesState {
            instances,
            instances_2d,
            animated_sprites,
            texts,
            deleted_texts,
            layout,
            tile_size,
            chunk_size,
            resources_path,
            default_texture_path,
            font,
            render_matrix
        }
    }

    /// Creates a new 3D model at the specifiec position.
    /// ### PANICS
    /// Will panic if the model's file is not found
    pub fn place_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
    ) -> InstanceReference {
        // TODO: Option to pass model instead of name of model
        let x = tile_position.0 * self.tile_size.0;
        let y = tile_position.1 * self.tile_size.1;
        let z = tile_position.2 * self.tile_size.2;
        self.place_model_absolute(model_name, gpu, (x, y, z))
    }

    fn place_model_absolute(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        (x, y, z): (f32, f32, f32),
    ) -> InstanceReference {
        match self.instances.contains_key(model_name) {
            true => {
                let instanced_m = self.instances.get_mut(model_name).unwrap();
                instanced_m.add_instance(x, y, z, &gpu.device, self.chunk_size);
            }
            false => {
                let model = load_glb_model(
                    model_name,
                    &gpu.device,
                    &gpu.queue,
                    &self.layout,
                    self.resources_path.clone(),
                    &self.default_texture_path,
                )
                .unwrap();
                let instanced_m = InstancedModel::new(model, &gpu.device, x, y, z, self.chunk_size);
                self.instances.insert(model_name.to_string(), instanced_m);
            }
        }

        let reference = InstanceReference {
            name: model_name.to_string(),
            index: self
                .instances
                .get(&model_name.to_string())
                .unwrap()
                .instances
                .len()
                - 1,
            dimension: Dimension::D3,
        };

        self.render_matrix.register_instance(reference.clone(), cgmath::vec3(x, y, z), self.instances.get(model_name).unwrap().model.get_extremes());

        reference
    }

    /// Creates a new 2D sprite at the specified position.
    /// All 2D sprites created from the same file will have the same "z" position. And cannot be changed once set.
    /// ### PANICS
    /// Will panic if the sprite's file is not found
    pub fn place_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
    ) -> InstanceReference {
        match self.instances_2d.contains_key(sprite_name) {
            true => {
                let instanced_s = self.instances_2d.get_mut(sprite_name).unwrap();
                instanced_s.add_instance(position.0, position.1, size, &gpu.device);
            }
            false => {
                let (sprite, width, height) = load_sprite(
                    sprite_name,
                    &gpu.device,
                    &gpu.queue,
                    &self.layout,
                    self.resources_path.clone(),
                )
                .unwrap();
                let (width, height) = match size {
                    Some((w, h)) => (w, h),
                    None => (width, height),
                };
                let instanced_s = InstancedSprite::new(
                    sprite,
                    &gpu.device,
                    position.0,
                    position.1,
                    position.2,
                    width,
                    height,
                );
                self.instances_2d
                    .insert(sprite_name.to_string(), instanced_s);
            }
        }

        InstanceReference {
            name: sprite_name.to_string(),
            index: self
                .instances_2d
                .get(&sprite_name.to_string())
                .unwrap()
                .instances
                .len()
                - 1,
            dimension: Dimension::D2,
        }
    }

    pub fn place_animated_sprite(
        &mut self,
        sprite_names: Vec<&str>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        frame_delay: std::time::Duration,
        looping: bool
    ) -> InstanceReference {
        let mut name = sprite_names.get(0).unwrap().to_string();
        while self.animated_sprites.contains_key(&name) {
            name += "A" ;
        }
        let sprites = sprite_names.into_iter()
            .map(|sprite_name|
                load_sprite(
                    sprite_name,
                    &gpu.device,
                    &gpu.queue,
                    &self.layout,
                    self.resources_path.clone(),
                )
                .unwrap()
            ).collect::<Vec<_>>();
        let (width, height) = (sprites.get(0).unwrap().1, sprites.get(0).unwrap().2);
        let sprites = sprites.into_iter().map(|(sprite, _, _)| sprite).collect();
        let (width, height) = match size {
            Some((w, h)) => (w, h),
            None => (width, height),
        };
        let instanced_s = AnimatedSprite::new(
            sprites,
            &gpu.device,
            position.0,
            position.1,
            position.2,
            width,
            height,
            frame_delay,
            looping
        );
        self.animated_sprites
            .insert(name.clone(), instanced_s);

        InstanceReference {
            name: name.clone(),
            index: self
                .instances_2d
                .get(&name)
                .unwrap()
                .instances
                .len()
                - 1,
            dimension: Dimension::Anim2D,
        }
    }

    /// Creates a new text at the specified position
    /// ### PANICS
    /// will panic if the characters' files are not found
    /// see: model::Font
    pub fn place_text(
        &mut self,
        text: Vec<String>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
    ) -> TextReference {
        let (text, w, h) = self.font.write_to_material(text, gpu, &self.layout);
        let instanced_t = match size {
            Some((w, h)) => {
                InstancedText::new(text, &gpu.device, position.0, position.1, position.2, w, h)
            }
            None => InstancedText::new(text, &gpu.device, position.0, position.1, position.2, w, h),
        };

        let index = match self.deleted_texts.pop() {
            Some(i) => {
                self.texts.push(Some(instanced_t));
                let last_item = self.texts.len() - 1;
                self.texts.swap(i, last_item);
                self.texts.pop();
                i
            }
            None => {
                self.texts.push(Some(instanced_t));
                self.texts.len() - 1
            }
        };

        TextReference { index }
    }

    /// Eliminates the text from screen and memory.
    pub fn forget_text(&mut self, text: TextReference) {
        self.texts.get_mut(text.index).unwrap().take();
        self.deleted_texts.push(text.index)
    }

    /// Saves all the 3D models' positions in a .temap file.
    pub fn save_temap(&self, file_name: &str, maps_path: String) {
        let mut map = temap::TeMap::new();
        for (name, instance) in &self.instances {
            map.add_model(&name);
            for inst in &instance.instances {
                map.add_instance(inst.position.x, inst.position.y, inst.position.z)
            }
        }

        map.save(&file_name, maps_path);
    }

    /// Load all 3D models from a .temap file.
    pub fn fill_from_temap(&mut self, map: temap::TeMap, gpu: &GpuState) {
        for (name, te_model) in map.models {
            for offset in te_model.offsets {
                self.place_model_absolute(&name, gpu, (offset.x, offset.y, offset.z));
            }
        }
    }

    /// Move a 3D model or a 2D sprite relative to its current position.
    /// Ignores z value on 2D sprites.
    pub fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &InstanceReference,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        match instance.dimension {
            Dimension::D2 => {
                let model = self.instances_2d.get_mut(&instance.name).unwrap();
                model.move_instance(instance.index, direction, queue);
            }
            Dimension::D3 => {
                let model = self.instances.get_mut(&instance.name).unwrap();
                self.render_matrix.unregister_instance(instance, model.instances.get(instance.index).unwrap().position, model.model.get_extremes());
                model.move_instance(instance.index, direction, queue);
                self.render_matrix.register_instance(instance.clone(), model.instances.get(instance.index).unwrap().position, model.model.get_extremes());
            }
            Dimension::Anim2D => {
                let model = self.animated_sprites.get_mut(&instance.name).unwrap();
                model.move_instance(instance.index, direction, queue);
            },
        };
    }

    /// Move a 3D model or a 2D sprite to an absolute position.
    /// Ignores z value on 2D sprites.
    pub fn set_instance_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &InstanceReference,
        position: P,
        queue: &wgpu::Queue,
    ) {
        match instance.dimension {
            Dimension::D2 => {
                let model = self.instances_2d.get_mut(&instance.name).unwrap();
                model.set_instance_position(instance.index, position, queue);
            }
            Dimension::D3 => {
                let model = self.instances.get_mut(&instance.name).unwrap();
                self.render_matrix.unregister_instance(instance, model.instances.get(instance.index).unwrap().position, model.model.get_extremes());
                model.set_instance_position(instance.index, position, queue);
                self.render_matrix.register_instance(instance.clone(), model.instances.get(instance.index).unwrap().position, model.model.get_extremes());
            }
            Dimension::Anim2D => {
                let model = self.animated_sprites.get_mut(&instance.name).unwrap();
                model.set_instance_position(instance.index, position, queue)
            },
        };
    }

    /// Get a 3D model's or 2D sprite's position.
    pub fn get_instance_position(&self, instance: &InstanceReference) -> (f32, f32, f32) {
        match instance.dimension {
            Dimension::D2 => {
                let sprite = self.instances_2d.get(&instance.name).unwrap();
                let position = sprite.instances.get(instance.index).unwrap().position;
                (position.x, position.y, sprite.depth)
            }
            Dimension::D3 => {
                let model = self.instances.get(&instance.name).unwrap();
                model.instances.get(instance.index).unwrap().position.into()
            }
            Dimension::Anim2D => {
                let sprite = self.animated_sprites.get(&instance.name).unwrap();
                let position = sprite.instance.position;
                (position.x, position.y, sprite.depth)
            },
        }
    }

    /// Changes the sprite's size. Using TODO algorithm
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn resize_sprite<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        instance: &InstanceReference,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        match instance.dimension {
            Dimension::D2 => {
                let sprite = self.instances_2d.get_mut(&instance.name).unwrap();
                sprite.resize(instance.index, new_size, queue);
            }
            Dimension::D3 => panic!("That is not a sprite"),
            Dimension::Anim2D => {
                let sprite = self.animated_sprites.get_mut(&instance.name).unwrap();
                sprite.resize(instance.index, new_size, queue);
            },
        };
    }

    /// Get the sprite's size
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn get_sprite_size(&self, instance: &InstanceReference) -> (f32, f32) {
        match instance.dimension {
            Dimension::D2 => {
                let sprite = self.get_sprite(instance);
                sprite.size.into()
            }
            Dimension::D3 => panic!("That is not a sprite"),
            Dimension::Anim2D => {
                let sprite = self.get_anim_sprite(instance);
                sprite.size.into()
            },
        }
    }

    /// Move a 2D text relative to it's current position.
    /// Ignores the z value.
    pub fn move_text<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &TextReference,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        let text = self
            .texts
            .get_mut(instance.index)
            .unwrap()
            .as_mut()
            .unwrap();
        text.move_instance(0, direction, queue);
    }

    /// Move a 2D text to an absolute position.
    /// Ignores the z value.
    pub fn set_text_position<P: Into<cgmath::Vector3<f32>>>(
        &mut self,
        instance: &TextReference,
        position: P,
        queue: &wgpu::Queue,
    ) {
        let text = self
            .texts
            .get_mut(instance.index)
            .unwrap()
            .as_mut()
            .unwrap();
        text.set_instance_position(0, position, queue);
    }

    /// Gets a 2D text's position
    pub fn get_text_position(&self, instance: &TextReference) -> (f32, f32) {
        let text = self.get_text(instance);
        text.position.into()
    }

    /// Resizes a 2D text
    pub fn resize_text<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        instance: &TextReference,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        let text = self
            .texts
            .get_mut(instance.index)
            .unwrap()
            .as_mut()
            .unwrap();
        text.resize(0, new_size, queue);
    }

    /// Gets a 2D text's size
    pub fn get_text_size(&self, instance: &TextReference) -> (f32, f32) {
        let text = self.get_text(instance);
        text.size.into()
    }

    pub fn set_instance_animation(&mut self, instance: &InstanceReference, animation: Animation) {
        match instance.dimension {
            Dimension::D2 => self.get_mut_sprite(instance).animation = Some(animation),
            Dimension::D3 => self.get_mut_model(instance).animation = Some(animation),
            Dimension::Anim2D => todo!(),
        }
    }

    pub fn set_text_animation(&mut self, text: &TextReference, animation: Animation) {
        self.get_mut_text(text).animation = Some(animation)
    }

    fn get_sprite(&self, instance: &InstanceReference) -> &Instance2D {
        let sprite = self.instances_2d.get(&instance.name).unwrap();
        sprite.instances.get(instance.index).unwrap()
    }

    fn _get_model(&self, instance: &InstanceReference) -> &Instance3D {
        let model = self.instances.get(&instance.name).unwrap();
        model.instances.get(instance.index).unwrap()
    }

    fn get_text(&self, text: &TextReference) -> &Instance2D {
        let text = self.texts.get(text.index).unwrap().as_ref().unwrap();
        &text.instance
    }

    fn get_mut_sprite(&mut self, instance: &InstanceReference) -> &mut Instance2D {
        let sprite = self.instances_2d.get_mut(&instance.name).unwrap();
        sprite.instances.get_mut(instance.index).unwrap()
    }

    pub(crate) fn get_mut_model(&mut self, instance: &InstanceReference) -> &mut Instance3D {
        let model = self.instances.get_mut(&instance.name).unwrap();
        model.instances.get_mut(instance.index).unwrap()
    }

    fn get_mut_text(&mut self, text: &TextReference) -> &mut Instance2D {
        let text = self.texts.get_mut(text.index).unwrap().as_mut().unwrap();
        &mut text.instance
    }

    pub(crate) fn mark_unculled(&mut self, instance: &mut InstanceReference, queue: &wgpu::Queue) {
        let model = self.instances.get_mut(&instance.name).unwrap();
        model.uncull_instance(queue, instance.index);
    }

    pub(crate) fn update_rendered(&mut self, frustum: &crate::camera::Frustum, queue: &wgpu::Queue) {
        for mut instance in self.render_matrix.update_rendered(frustum) {
            self.mark_unculled(&mut instance, queue)
        }
    }

    fn get_anim_sprite(&self, instance: &InstanceReference) -> &Instance2D {
        let sprite = self.animated_sprites.get(&instance.name).unwrap();
        &sprite.instance
    }
}

pub(crate) trait Draw2D {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, projection_bind_group: &'a wgpu::BindGroup, buffer: &'a wgpu::Buffer);
    fn get_depth(&self) -> f32;
}

impl Draw2D for InstancedText {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, projection_bind_group: &'a wgpu::BindGroup, buffer: &'a wgpu::Buffer) {
        use crate::model::DrawText;
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw_text(
            &self.image,
            projection_bind_group,
            buffer,
        );
    }

    fn get_depth(&self) -> f32 {
        self.depth
    }
}

impl Draw2D for InstancedSprite {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, projection_bind_group: &'a wgpu::BindGroup, buffer: &'a wgpu::Buffer) {
        use crate::model::DrawSprite;
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw_sprite_instanced(
            &self.sprite,
            0..self.instances.len() as u32,
            projection_bind_group,
            buffer,
        );
    }

    fn get_depth(&self) -> f32 {
        self.depth
    }
}

impl Draw2D for AnimatedSprite {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, projection_bind_group: &'a wgpu::BindGroup, buffer: &'a wgpu::Buffer) {
        use crate::model::DrawText;
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw_text(
            &self.get_sprite(),
            projection_bind_group,
            buffer,
        );
    }

    fn get_depth(&self) -> f32 {
        self.depth
    }
}
