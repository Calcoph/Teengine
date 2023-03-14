use std::ops::Range;

use wgpu::{RenderPass, ShaderStages};

use crate::{instances::{text::InstancedText, sprite::{InstancedSprite, AnimatedSprite}, InstanceReference, InstanceType}, model::{Model, Mesh, AnimatedModel, Material, AnimatedMesh}};


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
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.draw_sprite_instanced(
            &self.sprite,
            self.get_instances_vec(),
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

#[derive(Debug)]
pub struct InstanceFinder {
    markers: Vec<InstanceMarker>
}

impl InstanceFinder {
    pub(crate) fn new() -> InstanceFinder {
        InstanceFinder {
            markers: Vec::new()
        }
    }

    pub fn find_instance(&self, instance_count: u32) -> Option<InstanceReference> {
        if self.markers.len() == 0 || instance_count == 0 {
            return None
        }

        let mut search_range = 0..self.markers.len();
        let mut index = search_range.start + ((search_range.end - search_range.start) / 2);
        loop {
            let marker = match self.markers.get(index) {
                Some(m) => m,
                None => break,
            };

            match instance_count.cmp(&marker.start_instance_count) {
                std::cmp::Ordering::Less => {
                    search_range.end = index;
                    index = search_range.start + ((search_range.end - search_range.start) / 2);
                },
                std::cmp::Ordering::Equal => break,
                std::cmp::Ordering::Greater => {
                    let next_marker = match self.markers.get(index+1) {
                        Some(m) => m,
                        None => break,
                    };
                    match instance_count.cmp(&next_marker.start_instance_count) {
                        std::cmp::Ordering::Less => break, // instance is in `marker`
                        std::cmp::Ordering::Equal => { // instance is in `next_marker`
                            index = index + 1;
                            break
                        },
                        std::cmp::Ordering::Greater => {
                            search_range.start = index+1;
                            index = search_range.start + ((search_range.end - search_range.start) / 2);
                        }, // instance is not in `marker`
                    }
                },
            }

            if search_range.len() == 0 {
                break;
            }
        }

        match self.markers.get(index) {
            Some(marker) => {
                match instance_count.cmp(&marker.start_instance_count) {
                    std::cmp::Ordering::Less => None,
                    std::cmp::Ordering::Equal => Some(InstanceReference {
                        name: marker.model_name.clone(),
                        index: marker.start_instance_count as usize,
                        dimension: InstanceType::Opaque3D
                    }),
                    std::cmp::Ordering::Greater => match instance_count.cmp(&(marker.start_instance_count + marker.inst_range.len() as u32)) {
                        std::cmp::Ordering::Less => { // instance is in range
                            let index = (marker.inst_range.start + (instance_count - marker.start_instance_count)) as usize;
                            Some(InstanceReference {
                                name: marker.model_name.clone(),
                                index,
                                dimension: InstanceType::Opaque3D
                            })
                        },
                        std::cmp::Ordering::Equal => None, // instance out of range,
                        std::cmp::Ordering::Greater => None, // instance out of range
                    }
                }
            },
            None => None,
        }
    }
}

#[derive(Debug)]
struct InstanceMarker {
    start_instance_count: u32,
    inst_range: Range<u32>,
    model_name: String
}

pub struct RendererClickable<'a, 'b> {
    pub render_pass: &'b mut RenderPass<'a>,
    camera_bind_group: &'a wgpu::BindGroup,
    pub instance_count: u32,
    instance_finder: InstanceFinder
}

impl<'a, 'b, 'c> RendererClickable<'a, 'b>
    where 'c: 'a
{
    pub fn update_counter_constant(&mut self) {
        self.render_pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::cast_slice(&[self.instance_count]));
    }

    pub fn new(
        render_pass: &'b mut RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) -> RendererClickable<'a, 'b> {
        RendererClickable {
            render_pass,
            camera_bind_group,
            instance_count: 1, // 0 means there is no 
            instance_finder: InstanceFinder::new()
        }
    }

    pub fn get_instance_finder(self) -> InstanceFinder {
        self.instance_finder
    }

    pub fn draw_model_instanced_mask(
        &mut self,
        model: &'c Model,
        instances: Vec<Range<u32>>,
        model_name: String
    ) {
        for mesh in &model.meshes {
            self.update_counter_constant();
            self.draw_mesh_instanced_mask(mesh, instances.clone(), model_name.clone());
        }
    }
    fn draw_mesh_mask(
        &mut self,
        mesh: &'c Mesh,
        model_name: String
    ) {
        self.draw_mesh_instanced_mask(mesh, vec![0..1], model_name);
    }
    fn draw_animated_mesh_instanced_mask(
        &mut self,
        mesh: &'c AnimatedMesh,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        let range = 0..1;
        self.instance_count += range.len() as u32;
        self.render_pass.draw_indexed(0..mesh.num_elements, 0, range);
    }
    fn draw_mesh_instanced_mask(
        &mut self,
        mesh: &'c Mesh,
        instances: Vec<Range<u32>>,
        model_name: String
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        for inst_range in instances {
            self.update_counter_constant();
            self.instance_finder.markers.push(InstanceMarker {
                start_instance_count: self.instance_count,
                inst_range: inst_range.clone(),
                model_name: model_name.clone()
            });
            self.instance_count += inst_range.len() as u32;
            self.render_pass.draw_indexed(0..mesh.num_elements, 0, inst_range);
        }
    }
    pub fn draw_animated_model_instanced_mask(
        &mut self,
        model: &'c AnimatedModel,
    ) {
        for mesh in &model.meshes {
            self.update_counter_constant();
            self.draw_animated_mesh_instanced_mask(mesh);
        }
    }

    pub fn tdraw_model_instanced_mask(
        &mut self,
        model: &'c Model,
        instances: Vec<Range<u32>>,
    ) {
        for mesh in &model.transparent_meshes {
            self.tdraw_mesh_instanced_mask(mesh, &instances);
        }
    }
    pub fn tdraw_animated_model_instanced_mask(
        &mut self,
        model: &'c AnimatedModel,
    ) {
        for mesh in &model.meshes {
            self.tdraw_animated_mesh_instanced_mask(mesh);
        }
    }
    fn tdraw_mesh_mask(
        &mut self,
        mesh: &'c Mesh,
    ) {
        self.tdraw_mesh_instanced_mask(mesh, &vec![0..1]);
    }
    fn tdraw_animated_mesh_instanced_mask(
        &mut self,
        mesh: &'c AnimatedMesh,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        todo!(); // update counter
        self.render_pass.draw_indexed(0..mesh.num_elements, 0, 0..1);
    }

    fn tdraw_mesh_instanced_mask(
        &mut self,
        mesh: &'c Mesh,
        instances: &Vec<Range<u32>>,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        todo!(); // update counter
        for inst_range in instances {
            self.render_pass.draw_indexed(0..mesh.num_elements, 0, inst_range.clone());
        }
    }
}


pub struct Renderer<'a, 'b> {
    pub render_pass: &'b mut RenderPass<'a>,
    camera_bind_group: &'a wgpu::BindGroup,
}

impl<'a, 'b> Renderer<'a, 'b> {
    pub fn new(
        render_pass: &'b mut RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) -> Renderer<'a, 'b> {
        Renderer {
            render_pass,
            camera_bind_group,
        }
    }

    pub fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Vec<Range<u32>>,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone());
        }
    }

    pub fn draw_animated_model_instanced(
        &mut self,
        model: &'a AnimatedModel,
    ) {
        for mesh in &model.meshes {
            let material = mesh.materials[mesh.selected_material];
            let material = &model.materials[material];
            self.draw_animated_mesh_instanced(mesh, material);
        }
    }

    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
    ) {
        self.draw_mesh_instanced(mesh, material, vec![0..1]);
    }

    fn draw_animated_mesh_instanced(
        &mut self,
        mesh: &'a AnimatedMesh,
        material: &'a Material,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        self.render_pass.set_bind_group(1, &material.bind_group, &[]);
        self.render_pass.draw_indexed(0..mesh.num_elements, 0, 0..1);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Vec<Range<u32>>,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        self.render_pass.set_bind_group(1, &material.bind_group, &[]);
        for inst_range in instances {
            self.render_pass.draw_indexed(0..mesh.num_elements, 0, inst_range);
        }
    }

    pub fn tdraw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Vec<Range<u32>>,
    ) {
        for mesh in &model.transparent_meshes {
            let material = &model.materials[mesh.material];
            self.tdraw_mesh_instanced(mesh, material, &instances);
        }
    }

    pub fn tdraw_animated_model_instanced(
        &mut self,
        model: &'a AnimatedModel,
    ) {
        for mesh in &model.transparent_meshes {
            let material = mesh.materials[mesh.selected_material];
            let material = &model.materials[material];
            self.tdraw_animated_mesh_instanced(mesh, material);
        }
    }

    fn tdraw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
    ) {
        self.tdraw_mesh_instanced(mesh, material, &vec![0..1]);
    }

    fn tdraw_animated_mesh_instanced(
        &mut self,
        mesh: &'a AnimatedMesh,
        material: &'a Material,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        self.render_pass.set_bind_group(1, &material.bind_group, &[]);
        self.render_pass.draw_indexed(0..mesh.num_elements, 0, 0..1);
    }

    fn tdraw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: &Vec<Range<u32>>,
    ) {
        self.render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.render_pass.set_bind_group(0, self.camera_bind_group, &[]);
        self.render_pass.set_bind_group(1, &material.bind_group, &[]);
        for inst_range in instances {
            self.render_pass.draw_indexed(0..mesh.num_elements, 0, inst_range.clone());
        }
    }
}

pub trait DrawSprite<'a> {
    fn draw_sprite_instanced(
        &mut self,
        material: &'a Material,
        instances: Vec<Range<u32>>,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    );
}

impl<'a, 'b> DrawSprite<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_sprite_instanced(
        &mut self,
        material: &'a Material,
        instances: Vec<Range<u32>>,
        projection_bind_group: &'a wgpu::BindGroup,
        vertex_buffer: &'a wgpu::Buffer,
    ) {
        self.set_vertex_buffer(0, vertex_buffer.slice(..));
        self.set_bind_group(0, projection_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        for inst_range in instances {
            self.draw(0..6, inst_range);
        }
    }
}
