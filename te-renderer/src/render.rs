use std::ops::Range;

use crate::{instances::{text::InstancedText, sprite::{InstancedSprite, AnimatedSprite}}, model::{Model, Mesh, AnimatedModel, Material, AnimatedMesh}};


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

pub trait DrawModel<'a> {
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_animated_model_instanced(
        &mut self,
        model: &'a AnimatedModel,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_animated_mesh_instanced(
        &mut self,
        mesh: &'a AnimatedMesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), camera_bind_group);
        }
    }

    fn draw_animated_model_instanced(
        &mut self,
        model: &'b AnimatedModel,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = mesh.materials[mesh.selected_material];
            let material = &model.materials[material];
            self.draw_animated_mesh_instanced(mesh, material, camera_bind_group);
        }
    }

    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, vec![0..1], camera_bind_group);
    }

    fn draw_animated_mesh_instanced(
        &mut self,
        mesh: &'b AnimatedMesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, 0..1);
    }
    
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        for inst_range in instances {
            self.draw_indexed(0..mesh.num_elements, 0, inst_range);
        }
    }
}

pub trait DrawTransparentModel<'a> {
    fn tdraw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_animated_model_instanced(
        &mut self,
        model: &'a AnimatedModel,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_animated_mesh_instanced(
        &mut self,
        mesh: &'a AnimatedMesh,
        material: &'a Material,
        camera_bind_group: &'a wgpu::BindGroup,
    );
    fn tdraw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: &Vec<Range<u32>>,
        camera_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawTransparentModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn tdraw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Vec<Range<u32>>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.transparent_meshes {
            let material = &model.materials[mesh.material];
            self.tdraw_mesh_instanced(mesh, material, &instances, camera_bind_group);
        }
    }

    fn tdraw_animated_model_instanced(
        &mut self,
        model: &'b AnimatedModel,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.transparent_meshes {
            let material = mesh.materials[mesh.selected_material];
            let material = &model.materials[material];
            self.tdraw_animated_mesh_instanced(mesh, material, camera_bind_group);
        }
    }

    fn tdraw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.tdraw_mesh_instanced(mesh, material, &vec![0..1], camera_bind_group);
    }

    fn tdraw_animated_mesh_instanced(
        &mut self,
        mesh: &'b AnimatedMesh,
        material: &'b Material,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, 0..1);
    }

    fn tdraw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: &Vec<Range<u32>>,
        camera_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, &material.bind_group, &[]);
        for inst_range in instances {
            self.draw_indexed(0..mesh.num_elements, 0, inst_range.clone());
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
