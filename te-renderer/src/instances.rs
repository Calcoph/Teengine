use std::collections::{HashMap, HashSet};

use cgmath::Vector3;

pub mod animation;
pub mod model;
pub mod sprite;
pub mod text;
pub(crate) mod rangetree;

use crate::{
    resources::{load_glb_model, load_sprite},
    state::GpuState,
    temap, camera, model::{Model, AnimatedModel}, text::Font
};

use self::{animation::Animation, model::InstancedModel, sprite::{InstancedSprite, AnimatedSprite}, text::{InstancedText, TextReference}};

struct QuadTree {
    
}

#[derive(Debug)]
struct RenderMatrix {
    cells: Vec<Vec<Vec<InstanceReference>>>,
    cols: usize,
    chunk_size: (f32, f32, f32),
}

impl RenderMatrix {
    fn new(chunk_size: (f32, f32, f32)) -> Self {
        RenderMatrix {
            cells: Vec::new(),
            cols: 0,
            chunk_size,
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
                self.cells.push(Vec::new());
            };
            let row_vec = self.cells.get_mut(row).unwrap();
            while row_vec.len() <= col {
                row_vec.push(Vec::new());
            };
            row_vec.get_mut(col).unwrap().push(reference.clone());
            if col+1 > self.cols {
                self.cols = col+1;
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

    fn update_rendered(&mut self, view_cone: &camera::Frustum) -> Vec<&InstanceReference> {
        let mut viewed_chunks = Vec::new();
        for (i, cell) in self.cells.iter().enumerate() {
            for (j, references) in cell.iter().enumerate() {
                if view_cone.is_inside(i, j) {
                    viewed_chunks.extend(references.iter());
                }
            }
        }
        viewed_chunks
    }

    fn empty(&mut self) {
        self.cells = Vec::new();
        self.cols = 0;
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
    animation: Option<Animation>,
    hidden: bool,
    in_viewport: bool
}

impl Instance2D {
    fn new(position: cgmath::Vector2<f32>, size: cgmath::Vector2<f32>, animation: Option<Animation>, screen_w: u32, screen_h: u32) -> Self {
        let in_viewport = inside_viewport(position.into(), size.into(), (screen_w, screen_h));
        Instance2D {
            position,
            size,
            animation,
            hidden: false,
            in_viewport
        }
    }

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
            animation: None,
            hidden: self.hidden,
            in_viewport: self.in_viewport
        }
    }

    fn show(&mut self) {
        self.hidden = false
    }

    fn hide(&mut self) {
        self.hidden = true
    }

    fn is_hidden(&self) -> bool {
        self.hidden
    }

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

    #[must_use]
    fn move_direction<V: Into<cgmath::Vector3<f32>>>(&mut self, direction: V, screen_w: u32, screen_h: u32) -> Option<bool> {
        let direction = direction.into();
        self.position = self.position + cgmath::Vector2::new(direction.x, direction.y);
        let was_viewport = self.in_viewport;
        self.in_viewport = inside_viewport(self.position.into(), self.size.into(), (screen_w, screen_h));
        if was_viewport && !self.in_viewport && !self.hidden {
            Some(false)
        } else if !was_viewport && self.in_viewport && !self.hidden {
            Some(true)
        } else {
            None
        }
    }

    #[must_use]
    fn move_to<P: Into<cgmath::Vector3<f32>>>(&mut self, position: P, screen_w: u32, screen_h: u32) -> Option<bool> {
        let position = position.into();
        self.position = cgmath::Vector2::new(position.x, position.y);
        let was_viewport = self.in_viewport;
        self.in_viewport = inside_viewport(self.position.into(), self.size.into(), (screen_w, screen_h));
        if was_viewport && !self.in_viewport && !self.hidden {
            Some(false)
        } else if !was_viewport && self.in_viewport && !self.hidden {
            Some(true)
        } else {
            None
        }
    }
}

fn inside_viewport(
    (pos_x, pos_y): (f32, f32),
    (width, height): (f32, f32),
    (screen_w, screen_h): (u32, u32)
) -> bool {
    let pos_x = pos_x as i32;
    let pos_y = pos_y as i32;
    let width = width as i32;
    let height = height as i32;
    let screen_w = screen_w as i32;
    let screen_h = screen_h as i32;
    (
        // returns true if top-left is inside
        // +---------------+
        // |               |
        // |   .-.         |
        // |   | |      .----.
        // |   .-.      |  | |
        // +------------|--+ |
        //              .----.
        pos_x >= 0 && pos_x < screen_w
        && pos_y >= 0 && pos_y < screen_h
    ) || (
        // returns true if top-right is inside
        //   +---------------+
        //   |               |
        //   |               |
        //   |               |
        //.----.             |
        //|  +-|-------------+
        //.----.
        pos_x+width >= 0 && pos_x+width < screen_w
        && pos_y >= 0 && pos_y < screen_h
    ) || (
        // returns true if bot-left is inside
        //                .---.
        // +--------------|+  |
        // |              ||  |
        // |              .|--.
        // |               |
        // |               |
        // +---------------+
        pos_x >= 0 && pos_x < screen_w
        && pos_y+height >= 0 && pos_y+height < screen_h
    ) || (
        // returns true if bot-right is inside
        // .--------.
        // |  +-----|---------+
        // |  |     |         |
        // .------- .         |
        //    |               |
        //    |               |
        //    +---------------+
        pos_x+width >= 0 && pos_x+width < screen_w
        && pos_y+height >= 0 && pos_y+height < screen_h
    ) || (
        // returns true if no corner is inside, but region inside (vertical rectangles)
        // .---------------------.
        // |  +---------------+  |
        // |  |               |  |
        // |  |               |  |
        // |  |               |  |
        // |  |               |  |
        // |  +---------------+  |
        // |                     |
        // .---------------------.
        // .-------------.
        // |  +----------|----+
        // |  |          |    |
        // |  |          |    |
        // |  |          |    |
        // |  |          |    |
        // |  +----------|----+
        // |             |
        // .-------------.
        //           .-----------.
        //    +------|--------+  |
        //    |      |        |  |
        //    |      |        |  |
        //    |      |        |  |
        //    |      |        |  |
        //    +------|--------+  |
        //           |           |
        //           .-----------.
        pos_x < screen_w && pos_y < 0
        && pos_x+width >= 0 && pos_y+height >= screen_h
    ) || (
        // returns true if no corner is inside, but region inside (horizontal rectangles)
        // .---------------------.
        // |  +---------------+  |
        // |  |               |  |
        // |  |               |  |
        // .---------------------.
        //    |               |
        //    +---------------+
        //    +---------------+
        //    |               |
        // .---------------------.
        // |  |               |  |
        // |  |               |  |
        // |  +---------------+  |
        // |                     |
        // .---------------------.
        pos_x < 0 && pos_y < screen_h
        && pos_x+width >= screen_w && pos_y+height >= 0
    )
}

#[derive(Debug)]
pub struct Instance3D {
    pub position: cgmath::Vector3<f32>,
    pub animation: Option<Animation>,
    pub hidden: bool
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
            hidden: false
        }
    }

    pub(crate) fn hide(&mut self) {
        self.hidden = true
    }

    pub(crate) fn show(&mut self) {
        self.hidden = false
    }

    pub(crate) fn is_hidden(&self) -> bool {
        self.hidden
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

pub trait InstancedDraw {
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
enum InstanceType {
    Sprite,
    Anim2D,
    Opaque3D,
}

/// Handle of a 3D model or 2D sprite. You will need it when changing their properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceReference {
    name: String,
    index: usize,
    dimension: InstanceType,
}

impl InstanceReference {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_id(&self) -> usize {
        self.index
    }
}

#[derive(Debug)]
pub enum DrawModel {
    M(InstancedModel),
    A(AnimatedModel)
}

impl DrawModel {
    fn get_mut_m(&mut self) -> &mut InstancedModel {
        match self {
            DrawModel::M(m) => m,
            DrawModel::A(_) => panic!("DrawModel unwrap failed"),
        }
    }

    fn get_m(&self) -> &InstancedModel {
        match self {
            DrawModel::M(m) => m,
            DrawModel::A(_) => panic!("DrawModel unwrap failed"),
        }
    }

    fn get_mut_a(&mut self) -> &mut AnimatedModel {
        match self {
            DrawModel::M(_) => panic!("DrawModel unwrap failed"),
            DrawModel::A(a) => a,
        }
    }

    fn get_a(&self) -> &AnimatedModel {
        match self {
            DrawModel::M(_) => panic!("DrawModel unwrap failed"),
            DrawModel::A(a) => a,
        }
    }

    fn hide(&mut self, index: usize) {
        match self {
            DrawModel::M(dm) => dm.hide(index),
            DrawModel::A(dm) => dm.hide(),
        }
    }

    fn show(&mut self, index: usize) {
        match self {
            DrawModel::M(dm) => dm.show(index),
            DrawModel::A(dm) => dm.show(),
        }
    }

    fn is_hidden(&self, index: usize) -> bool {
        match self {
            DrawModel::M(dm) => dm.is_hidden(index),
            DrawModel::A(dm) => dm.is_hidden(),
        }
    }
}

/// Manages the window's 3D models, 2D sprites and 2D texts
#[derive(Debug)]
pub struct InstancesState {
    pub(crate) opaque_instances: HashMap<String, DrawModel>,
    pub(crate) transparent_instances: HashSet<String>,
    pub(crate) sprite_instances: HashMap<String, InstancedSprite>,
    pub(crate) animated_sprites: HashMap<String, AnimatedSprite>,
    pub(crate) texts: Vec<Option<InstancedText>>,
    deleted_texts: Vec<usize>,
    pub layout: wgpu::BindGroupLayout,
    tile_size: (f32, f32, f32),
    pub resources_path: String,
    pub default_texture_path: String,
    font: Font,
    render_matrix: RenderMatrix
}

impl InstancesState {
    pub(crate) fn new(
        layout: wgpu::BindGroupLayout,
        tile_size: (f32, f32, f32),
        chunk_size: (f32, f32, f32),
        resources_path: String,
        default_texture_path: String,
        font_dir_path: String,
    ) -> Self {
        let opaque_instances = HashMap::new();
        let transparent_instances = HashSet::new();
        let sprite_instances = HashMap::new();
        let animated_sprites = HashMap::new();
        let texts = Vec::new();
        let deleted_texts = Vec::new();
        let font = Font::new(font_dir_path);
        let render_matrix = RenderMatrix::new(chunk_size);

        InstancesState {
            opaque_instances,
            transparent_instances,
            sprite_instances,
            animated_sprites,
            texts,
            deleted_texts,
            layout,
            tile_size,
            resources_path,
            default_texture_path,
            font,
            render_matrix
        }
    }

    /// Creates a new 3D model at the specific position.
    /// ### PANICS
    /// Will panic if the model's file is not found
    pub(crate) fn place_model(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
    ) -> InstanceReference {
        let x = tile_position.0 * self.tile_size.0;
        let y = tile_position.1 * self.tile_size.1;
        let z = tile_position.2 * self.tile_size.2;
        self.place_model_absolute(model_name, gpu, (x, y, z))
    }

    /// Places an already created model at the specific position.
    /// If that model has not been forgotten, you can place another with just its name, so model can be None
    /// ### PANICS
    /// Will panic if model is None and the model has been forgotten (or was never created)
    pub(crate) fn place_custom_model (
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        model: Option<Model>
    ) -> InstanceReference {
        let x = tile_position.0 * self.tile_size.0;
        let y = tile_position.1 * self.tile_size.1;
        let z = tile_position.2 * self.tile_size.2;
        self.place_custom_model_absolute(model_name, gpu, (x, y, z), model)
    }

    fn place_custom_model_absolute (
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        (x, y, z): (f32, f32, f32),
        model: Option<Model>
    ) -> InstanceReference {
        match self.opaque_instances.contains_key(model_name) {
            true => {
                let instanced_m = self.opaque_instances.get_mut(model_name).unwrap();
                match instanced_m {
                    DrawModel::M(m) => m.add_instance(x, y, z, &gpu.device),
                    DrawModel::A(_) => panic!("This is impossible"),
                }
            }
            false => {
                let model = model.unwrap();
                let transparent_meshes = model.transparent_meshes.len();
                let instanced_m = InstancedModel::new(model, &gpu.device, x, y, z);
                self.opaque_instances.insert(model_name.to_string(), DrawModel::M(instanced_m));
                if transparent_meshes > 0 {
                    self.transparent_instances.insert(model_name.to_string());
                }
            }
        }
        let index = self.opaque_instances.get(&model_name.to_string())
            .unwrap()
            .get_m()
            .instances
            .len()-1;

        let reference = InstanceReference {
            name: model_name.to_string(),
            index,
            dimension: InstanceType::Opaque3D,
        };

        self.render_matrix.register_instance(
            reference.clone(),
            cgmath::vec3(x, y, z),
            self.opaque_instances.get(model_name)
                .unwrap()
                .get_m()
                .model
                .get_extremes()
        );

        reference
    }

    /// Places an already created animated model at the specific position.
    pub(crate) fn place_custom_animated_model (
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        tile_position: (f32, f32, f32),
        mut model: AnimatedModel
    ) -> InstanceReference {
        let x = tile_position.0 * self.tile_size.0;
        let y = tile_position.1 * self.tile_size.1;
        let z = tile_position.2 * self.tile_size.2;
        model.set_instance_position(0, Vector3::new(x, y, z), &gpu.queue);
        self.place_custom_animated_model_absolute(model_name, model)
    }

    fn place_custom_animated_model_absolute (
        &mut self,
        model_name: &str,
        model: AnimatedModel
    ) -> InstanceReference {
        let transparent_meshes = model.transparent_meshes.len();
        let position = model.instance.position;
        self.opaque_instances.insert(model_name.to_string(), DrawModel::A(model));
        if transparent_meshes > 0 {
            self.transparent_instances.insert(model_name.to_string());
        }

        let reference = InstanceReference {
            name: model_name.to_string(),
            index: 0,
            dimension: InstanceType::Opaque3D,
        };

        self.render_matrix.register_instance(
            reference.clone(),
            position,
            self.opaque_instances.get(model_name)
                .unwrap()
                .get_a()
                .get_extremes()
        );

        reference
    }


    fn place_model_absolute(
        &mut self,
        model_name: &str,
        gpu: &GpuState,
        (x, y, z): (f32, f32, f32),
    ) -> InstanceReference {
        match self.opaque_instances.contains_key(model_name) {
            true => {
                let instanced_m = self.opaque_instances.get_mut(model_name).unwrap().get_mut_m();
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
                let transparent_meshes = model.transparent_meshes.len();
                let instanced_m = InstancedModel::new(model, &gpu.device, x, y, z);
                self.opaque_instances.insert(model_name.to_string(), DrawModel::M(instanced_m));
                if transparent_meshes > 0 {
                    self.transparent_instances.insert(model_name.to_string());
                }
            }
        }

        let reference = InstanceReference {
            name: model_name.to_string(),
            index: self
                .opaque_instances
                .get(&model_name.to_string())
                .unwrap()
                .get_m()
                .instances
                .len()
                - 1,
            dimension: InstanceType::Opaque3D,
        };

        self.render_matrix.register_instance(
            reference.clone(),
            cgmath::vec3(x, y, z),
            self.opaque_instances.get(model_name)
                .unwrap()
                .get_m()
                .model
                .get_extremes()
        );

        reference
    }

    /// Creates a new 2D sprite at the specified position.
    /// All 2D sprites created from the same file will have the same "z" position. And cannot be changed once set.
    /// ### PANICS
    /// Will panic if the sprite's file is not found
    pub(crate) fn place_sprite(
        &mut self,
        sprite_name: &str,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        screen_w: u32,
        screen_h: u32
    ) -> InstanceReference {
        match self.sprite_instances.contains_key(sprite_name) {
            true => {
                let instanced_s = self.sprite_instances.get_mut(sprite_name).unwrap();
                instanced_s.add_instance(
                    position.0,
                    position.1,
                    size,
                    &gpu.device,
                    screen_w,
                    screen_h
                );
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
                    screen_w,
                    screen_h
                );
                self.sprite_instances
                    .insert(sprite_name.to_string(), instanced_s);
            }
        }

        InstanceReference {
            name: sprite_name.to_string(),
            index: self
                .sprite_instances
                .get(&sprite_name.to_string())
                .unwrap()
                .instances
                .len()
                - 1,
            dimension: InstanceType::Sprite,
        }
    }

    pub(crate) fn place_animated_sprite(
        &mut self,
        sprite_names: Vec<&str>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        frame_delay: std::time::Duration,
        looping: bool,
        screen_w: u32,
        screen_h: u32
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
            looping,
            screen_w,
            screen_h
        );
        self.animated_sprites
            .insert(name.clone(), instanced_s);

        InstanceReference {
            name,
            index: 0,
            dimension: InstanceType::Anim2D,
        }
    }

    /// Creates a new text at the specified position
    /// ### PANICS
    /// will panic if the characters' files are not found
    /// see: model::Font
    pub(crate) fn place_text(
        &mut self,
        text: Vec<String>,
        gpu: &GpuState,
        size: Option<(f32, f32)>,
        position: (f32, f32, f32),
        screen_w: u32,
        screen_h: u32
    ) -> TextReference {
        let (text, w, h) = self.font.write_to_material(text, gpu, &self.layout);
        let instanced_t = match size {
            Some((w, h)) => {
                InstancedText::new(text, &gpu.device, position.0, position.1, position.2, w, h, screen_w, screen_h)
            }
            None => InstancedText::new(text, &gpu.device, position.0, position.1, position.2, w, h, screen_w, screen_h),
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
    pub(crate) fn forget_text(&mut self, text: TextReference) {
        self.texts.get_mut(text.index).unwrap().take();
        self.deleted_texts.push(text.index)
    }

    pub(crate) fn forget_all_2d_instances(&mut self) {
        self.animated_sprites = HashMap::new();
        self.sprite_instances = HashMap::new();
    }

    pub(crate) fn forget_all_3d_instances(&mut self) {
        self.opaque_instances = HashMap::new();
        self.transparent_instances = HashSet::new();
        self.render_matrix.empty();
    }

    pub(crate) fn forget_all_instances(&mut self) {
        self.animated_sprites = HashMap::new();
        self.sprite_instances = HashMap::new();
        self.opaque_instances = HashMap::new();
        self.transparent_instances = HashSet::new();
        self.render_matrix.empty();
    }

    /// Saves all the 3D models' positions in a .temap file.
    pub(crate) fn save_temap(&self, file_name: &str, maps_path: String) {
        // TODO: Maybe get maps_path from initial_configuration
        let mut map = temap::TeMap::new();
        for (name, instance) in &self.opaque_instances {
            map.add_model(&name);
            match instance {
                DrawModel::M(m) => {
                    for inst in m.instances.iter() {
                        map.add_instance(inst.position.x, inst.position.y, inst.position.z)
                    }
                },
                DrawModel::A(a) => map.add_instance(a.instance.position.x, a.instance.position.y, a.instance.position.z),
            }
        }

        map.save(&file_name, maps_path);
    }

    /// Load all 3D models from a .temap file.
    pub(crate) fn fill_from_temap(&mut self, map: temap::TeMap, gpu: &GpuState) {
        for (name, te_model) in map.models {
            for offset in te_model.offsets {
                self.place_model_absolute(&name, gpu, (offset.x, offset.y, offset.z));
            }
        }
    }

    /// Move a 3D model or a 2D sprite relative to its current position.
    /// Ignores z value on 2D sprites.
    pub(crate) fn move_instance(
        &mut self,
        instance: &InstanceReference,
        direction: cgmath::Vector3<f32>,
        queue: &wgpu::Queue,
        screen_w: u32,
        screen_h: u32
    ) {
        match instance.dimension {
            InstanceType::Sprite => {
                let model = self.sprite_instances.get_mut(&instance.name).unwrap();
                model.move_instance(instance.index, direction, queue, screen_w, screen_h);
            }
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get_mut(&instance.name).unwrap();
                match model {
                    DrawModel::M(m) => {
                        self.render_matrix.unregister_instance(instance, m.instances.get(instance.index).unwrap().position, m.model.get_extremes());
                        m.move_instance(instance.index, direction, queue);
                        self.render_matrix.register_instance(instance.clone(), m.instances.get(instance.index).unwrap().position, m.model.get_extremes());      
                    },
                    DrawModel::A(a) => {
                        self.render_matrix.unregister_instance(instance, a.instance.position, a.get_extremes());
                        a.move_instance(instance.index, direction, queue);
                        self.render_matrix.register_instance(instance.clone(), a.instance.position, a.get_extremes());
                    },
                };
            }
            InstanceType::Anim2D => {
                let model = self.animated_sprites.get_mut(&instance.name).unwrap();
                model.move_instance(instance.index, direction, queue, screen_w, screen_h);
            },
        };
    }

    /// Move a 3D model or a 2D sprite to an absolute position.
    /// Ignores z value on 2D sprites.
    pub(crate) fn set_instance_position(
        &mut self,
        instance: &InstanceReference,
        position: cgmath::Vector3<f32>,
        queue: &wgpu::Queue,
        screen_w: u32,
        screen_h: u32
    ) {
        match instance.dimension {
            InstanceType::Sprite => {
                let model = self.sprite_instances.get_mut(&instance.name).unwrap();
                model.set_instance_position(instance.index, position, queue, screen_w, screen_h);
            }
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get_mut(&instance.name).unwrap();
                match model {
                    DrawModel::M(m) => {
                        self.render_matrix.unregister_instance(instance, m.instances.get(instance.index).unwrap().position, m.model.get_extremes());
                        m.set_instance_position(instance.index, position, queue);
                        self.render_matrix.register_instance(instance.clone(), m.instances.get(instance.index).unwrap().position, m.model.get_extremes());
                    },
                    DrawModel::A(a) => {
                        self.render_matrix.unregister_instance(instance, a.instance.position, a.get_extremes());
                        a.set_instance_position(instance.index, position, queue);
                        self.render_matrix.register_instance(instance.clone(), a.instance.position, a.get_extremes());
                    },
                };
            }
            InstanceType::Anim2D => {
                let model = self.animated_sprites.get_mut(&instance.name).unwrap();
                model.set_instance_position(instance.index, position, queue, screen_w, screen_h)
            },
        };
    }

    /// Get a 3D model's or 2D sprite's position.
    pub(crate) fn get_instance_position(&self, instance: &InstanceReference) -> (f32, f32, f32) {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.sprite_instances.get(&instance.name).unwrap();
                let position = sprite.instances.get(instance.index).unwrap().position;
                (position.x, position.y, sprite.depth)
            }
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get(&instance.name).unwrap();
                match model {
                    DrawModel::M(m) => m.instances.get(instance.index).unwrap().position.into(),
                    DrawModel::A(a) => a.instance.position.into(),
                }
            }
            InstanceType::Anim2D => {
                let sprite = self.animated_sprites.get(&instance.name).unwrap();
                let position = sprite.instance.position;
                (position.x, position.y, sprite.depth)
            },
        }
    }

    /// Changes the sprite's size. Using TODO algorithm
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub(crate) fn resize_sprite(
        &mut self,
        instance: &InstanceReference,
        new_size: cgmath::Vector2<f32>,
        queue: &wgpu::Queue,
    ) {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.sprite_instances.get_mut(&instance.name).unwrap();
                sprite.resize(instance.index, new_size, queue);
            }
            InstanceType::Opaque3D => panic!("That is not a sprite"),
            InstanceType::Anim2D => {
                let sprite = self.animated_sprites.get_mut(&instance.name).unwrap();
                sprite.resize(instance.index, new_size, queue);
            },
        };
    }

    /// Get the sprite's size
    /// ### PANICS
    /// Will panic if a 3D model's reference is passed instead of a 2D sprite's.
    pub(crate) fn get_sprite_size(&self, instance: &InstanceReference) -> (f32, f32) {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.get_sprite(instance);
                sprite.size.into()
            }
            InstanceType::Opaque3D => panic!("That is not a sprite"),
            InstanceType::Anim2D => {
                let sprite = self.get_anim_sprite(instance);
                sprite.size.into()
            },
        }
    }

    /// Move a 2D text relative to it's current position.
    /// Ignores the z value.
    pub(crate) fn move_text(
        &mut self,
        instance: &TextReference,
        direction: cgmath::Vector3<f32>,
        queue: &wgpu::Queue,
        screen_w: u32,
        screen_h: u32
    ) {
        let text = self
            .texts
            .get_mut(instance.index)
            .unwrap()
            .as_mut()
            .unwrap();
        text.move_instance(0, direction, queue, screen_w, screen_h);
    }

    /// Move a 2D text to an absolute position.
    /// Ignores the z value.
    pub(crate) fn set_text_position(
        &mut self,
        instance: &TextReference,
        position: cgmath::Vector3<f32>,
        queue: &wgpu::Queue,
        screen_w: u32,
        screen_h: u32
    ) {
        let text = self
            .texts
            .get_mut(instance.index)
            .unwrap()
            .as_mut()
            .unwrap();
        text.set_instance_position(0, position, queue, screen_w, screen_h);
    }

    /// Gets a 2D text's position
    pub(crate) fn get_text_position(&self, instance: &TextReference) -> (f32, f32) {
        let text = self.get_text(instance);
        text.position.into()
    }

    /// Resizes a 2D text
    pub(crate) fn resize_text(
        &mut self,
        instance: &TextReference,
        new_size: cgmath::Vector2<f32>,
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
    pub(crate) fn get_text_size(&self, instance: &TextReference) -> (f32, f32) {
        let text = self.get_text(instance);
        text.size.into()
    }

    pub(crate) fn set_instance_animation(&mut self, instance: &InstanceReference, animation: Animation) {
        match instance.dimension {
            InstanceType::Sprite => self.get_mut_sprite(instance).animation = Some(animation),
            InstanceType::Opaque3D => self.get_mut_model(instance).animation = Some(animation),
            InstanceType::Anim2D => todo!(),
        }
    }

    pub(crate) fn set_text_animation(&mut self, text: &TextReference, animation: Animation) {
        self.get_mut_text(text).animation = Some(animation)
    }

    fn get_sprite(&self, instance: &InstanceReference) -> &Instance2D {
        let sprite = self.sprite_instances.get(&instance.name).unwrap();
        sprite.instances.get(instance.index).unwrap()
    }

    fn _get_model(&self, instance: &InstanceReference) -> &Instance3D {
        let model = self.opaque_instances.get(&instance.name).unwrap();
        match model {
            DrawModel::M(m) => m.instances.get(instance.index).unwrap(),
            DrawModel::A(a) => &a.instance,
        }
    }

    fn get_text(&self, text: &TextReference) -> &Instance2D {
        let text = self.texts.get(text.index).unwrap().as_ref().unwrap();
        &text.instance
    }

    fn get_mut_sprite(&mut self, instance: &InstanceReference) -> &mut Instance2D {
        let sprite = self.sprite_instances.get_mut(&instance.name).unwrap();
        sprite.instances.get_mut(instance.index).unwrap()
    }

    pub(crate) fn get_mut_model(&mut self, instance: &InstanceReference) -> &mut Instance3D {
        let model = self.opaque_instances.get_mut(&instance.name).unwrap();
        match model {
            DrawModel::M(m) => m.instances.get_mut(instance.index).unwrap(),
            DrawModel::A(a) => &mut a.instance,
        }
    }

    fn get_mut_text(&mut self, text: &TextReference) -> &mut Instance2D {
        let text = self.texts.get_mut(text.index).unwrap().as_mut().unwrap();
        &mut text.instance
    }

    pub(crate) fn update_rendered3d(&mut self, frustum: &crate::camera::Frustum) {
        for instance in self.render_matrix.update_rendered(frustum) {
            let model = self.opaque_instances.get_mut(&instance.name).unwrap();
            match model {
                DrawModel::M(m) => m.uncull_instance(instance.index),
                DrawModel::A(a) => a.uncull_instance(),
            }
        }
    }

    fn get_anim_sprite(&self, instance: &InstanceReference) -> &Instance2D {
        let sprite = self.animated_sprites.get(&instance.name).unwrap();
        &sprite.instance
    }

    pub(crate) fn animate_model(&mut self, instance: &InstanceReference, mesh_index: usize, material_index: usize) {
        match instance.dimension {
            InstanceType::Sprite => (),
            InstanceType::Anim2D => (),
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get_mut(instance.get_name());
                match model {
                    Some(mo) => match mo {
                        DrawModel::M(_) => (),
                        DrawModel::A(a) => a.animate(mesh_index, material_index),
                    },
                    None => ()
                }

            },
        }
    }

    pub(crate) fn hide_instance(&mut self, instance: &InstanceReference) {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.sprite_instances.get_mut(instance.get_name());
                sprite.unwrap().hide(instance.get_id());
            },
            InstanceType::Anim2D => {
                let anim = self.animated_sprites.get_mut(instance.get_name());
                anim.unwrap().hide();
            },
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get_mut(instance.get_name());
                model.unwrap().hide(instance.get_id());
            },
        }
    }

    pub(crate) fn show_instance(&mut self, instance: &InstanceReference) {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.sprite_instances.get_mut(instance.get_name());
                sprite.unwrap().show(instance.get_id());
            },
            InstanceType::Anim2D => {
                let anim = self.animated_sprites.get_mut(instance.get_name());
                anim.unwrap().show();
            },
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get_mut(instance.get_name());
                model.unwrap().show(instance.get_id());
            },
        }
    }

    pub(crate) fn is_hidden(&self, instance: &InstanceReference) -> bool {
        match instance.dimension {
            InstanceType::Sprite => {
                let sprite = self.sprite_instances.get(instance.get_name());
                sprite.unwrap().is_hidden(instance.get_id())
            },
            InstanceType::Anim2D => {
                let anim = self.animated_sprites.get(instance.get_name());
                anim.unwrap().is_hidden()
            },
            InstanceType::Opaque3D => {
                let model = self.opaque_instances.get(instance.get_name());
                model.unwrap().is_hidden(instance.get_id())
            },
        }
    }
}
