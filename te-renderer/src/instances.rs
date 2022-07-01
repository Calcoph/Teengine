use std::collections::HashMap;

use cgmath::Vector3;
use wgpu::util::DeviceExt;
use winit::{dpi, window::Window};

use crate::{
    model,
    resources::{load_glb_model, load_sprite},
    state::GpuState,
    temap, texture, animation::Animation,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl model::Vertex for InstanceRaw {
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
}

impl Instance for Instance3D {
    fn to_raw(&mut self) -> InstanceRaw {
        let model = cgmath::Matrix4::from_translation(self.position);
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

#[derive(Debug)]
pub struct InstancedModel {
    pub model: model::Model,
    pub instances: Vec<Instance3D>,
    pub instance_buffer: wgpu::Buffer,
}

impl InstancedModel {
    fn new(model: model::Model, device: &wgpu::Device, x: f32, y: f32, z: f32) -> Self {
        let instances = vec![Instance3D {
            position: cgmath::Vector3 { x, y, z },
        }];

        InstancedModel::new_premade(model, device, instances)
    }

    fn new_premade(model: model::Model, device: &wgpu::Device, mut instances: Vec<Instance3D>) -> Self {
        let instance_data = instances.iter_mut().map(Instance3D::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedModel {
            model,
            instances,
            instance_buffer,
        }
    }

    fn add_instance(&mut self, x: f32, y: f32, z: f32, device: &wgpu::Device) {
        let new_instance = Instance3D {
            position: cgmath::Vector3 { x, y, z },
        };
        //TODO: see if there is a better way than replacing the buffer with a new one
        self.instances.push(new_instance);
        self.instance_buffer.destroy();
        let instance_data = self
            .instances
            .iter_mut()
            .map(|instance| instance.to_raw())
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.instance_buffer = instance_buffer;
    }
}

impl InstancedDraw for InstancedModel {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_direction(direction);
        let raw = instance.to_raw();
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
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_to(position);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }
}

#[derive(Debug)]
pub struct InstancedSprite {
    pub sprite: model::Material,
    width: f32,
    height: f32,
    pub instances: Vec<Instance2D>,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
}

impl InstancedSprite {
    fn new(
        sprite: model::Material,
        device: &wgpu::Device,
        x: f32,
        y: f32,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let instances = vec![Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
        }];

        InstancedSprite::new_premade(sprite, device, instances, depth, w, h)
    }

    fn new_premade(
        sprite: model::Material,
        device: &wgpu::Device,
        mut instances: Vec<Instance2D>,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let instance_data = instances.iter_mut().map(Instance2D::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedSprite {
            sprite,
            width: w,
            height: h,
            instances,
            instance_buffer,
            depth,
        }
    }

    fn add_instance(&mut self, x: f32, y: f32, size: Option<(f32, f32)>, device: &wgpu::Device) {
        let (w, h) = match size {
            Some((width, height)) => (width, height),
            None => (self.width, self.height),
        };
        let new_instance = Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
        };
        //TODO: see if there is a better way than replacing the buffer with a new one
        self.instances.push(new_instance);
        self.instance_buffer.destroy();
        let instance_data = self
            .instances
            .iter_mut()
            .map(|instance| instance.to_raw())
            .collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.instance_buffer = instance_buffer;
    }

    fn resize<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        index: usize,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.resize(new_size);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }
}

impl InstancedDraw for InstancedSprite {
    fn move_instance<V: Into<cgmath::Vector3<f32>>>(
        &mut self,
        index: usize,
        direction: V,
        queue: &wgpu::Queue,
    ) {
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_direction(direction);
        let raw = instance.to_raw();
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
        let instance = self.instances.get_mut(index).unwrap();
        instance.move_to(position);
        let raw = instance.to_raw();
        queue.write_buffer(
            &self.instance_buffer,
            (index * std::mem::size_of::<InstanceRaw>())
                .try_into()
                .unwrap(),
            bytemuck::cast_slice(&[raw]),
        );
    }
}

#[derive(Debug)]
pub struct InstancedText {
    pub image: model::Material,
    pub instance: Instance2D,
    pub instance_buffer: wgpu::Buffer,
    pub depth: f32,
}

impl InstancedText {
    fn new(
        image: model::Material,
        device: &wgpu::Device,
        x: f32,
        y: f32,
        depth: f32,
        w: f32,
        h: f32,
    ) -> Self {
        let mut instance = Instance2D {
            position: cgmath::Vector2 { x, y },
            size: cgmath::Vector2 { x: w, y: h },
            animation: None
        };

        let instance_data = [instance.to_raw()];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        InstancedText {
            image,
            instance,
            instance_buffer,
            depth,
        }
    }

    fn resize<V: Into<cgmath::Vector2<f32>>>(
        &mut self,
        index: usize,
        new_size: V,
        queue: &wgpu::Queue,
    ) {
        self.instance.resize(new_size);
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

impl InstancedDraw for InstancedText {
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

enum Dimension {
    D2,
    D3,
}

/// Handle of a 3D model or 2D sprite. You will need it when changing their properties.
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

/// Handle of a 2D text. You will need it when changing its properties.
pub struct TextReference {
    index: usize,
}

/// Manages the window's 3D models, 2D sprites and 2D texts
#[derive(Debug)]
pub struct InstancesState {
    pub instances: HashMap<String, InstancedModel>,
    pub instances_2d: HashMap<String, InstancedSprite>,
    pub texts: Vec<Option<InstancedText>>,
    deleted_texts: Vec<usize>,
    pub layout: wgpu::BindGroupLayout,
    tile_size: (f32, f32, f32),
    pub resources_path: String,
    pub default_texture_path: String,
    font: model::Font,
}

impl InstancesState {
    pub fn new(
        layout: wgpu::BindGroupLayout,
        tile_size: (f32, f32, f32),
        resources_path: String,
        default_texture_path: String,
        font_dir_path: String,
    ) -> Self {
        let instances = HashMap::new();
        let instances_2d = HashMap::new();
        let texts = Vec::new();
        let deleted_texts = Vec::new();
        let font = model::Font::new(font_dir_path);

        InstancesState {
            instances,
            instances_2d,
            texts,
            deleted_texts,
            layout,
            tile_size,
            resources_path,
            default_texture_path,
            font,
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
        match self.instances.contains_key(model_name) {
            true => {
                let instanced_m = self.instances.get_mut(model_name).unwrap();
                instanced_m.add_instance(x, y, z, &gpu.device);
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
                let instanced_m = InstancedModel::new(model, &gpu.device, x, y, z);
                self.instances.insert(model_name.to_string(), instanced_m);
            }
        }

        InstanceReference {
            name: model_name.to_string(),
            index: self
                .instances
                .get(&model_name.to_string())
                .unwrap()
                .instances
                .len()
                - 1,
            dimension: Dimension::D3,
        }
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
            let model = load_glb_model(
                &name,
                &gpu.device,
                &gpu.queue,
                &self.layout,
                self.resources_path.clone(),
                &self.default_texture_path,
            );
            match model {
                Ok(m) => {
                    let mut instance_vec = Vec::new();
                    for offset in te_model.offsets {
                        let instance = Instance3D {
                            position: cgmath::Vector3 {
                                x: offset.x,
                                y: offset.y,
                                z: offset.z,
                            },
                        };
                        instance_vec.push(instance);
                    }
                    self.instances.insert(
                        name,
                        InstancedModel::new_premade(m, &gpu.device, instance_vec),
                    );
                }
                _ => (),
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
                model.move_instance(instance.index, direction, queue);
            }
        };
    }

    /// Move a 3D model or a 2D sprite to an absolute position.
    /// /// Ignores z value on 2D sprites.
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
                model.set_instance_position(instance.index, position, queue);
            }
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
        };
    }

    /// Get the sprite's size
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub fn get_sprite_size(&self, instance: &InstanceReference) -> (f32, f32) {
        match instance.dimension {
            Dimension::D2 => {
                let sprite = self.instances_2d.get(&instance.name).unwrap();
                sprite.instances.get(instance.index).unwrap().size.into()
            }
            Dimension::D3 => panic!("That is not a sprite"),
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
        let text = self.texts.get(instance.index).unwrap().as_ref().unwrap();
        text.instance.position.into()
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
        let text = self.texts.get(instance.index).unwrap().as_ref().unwrap();
        text.instance.size.into()
    }
}
