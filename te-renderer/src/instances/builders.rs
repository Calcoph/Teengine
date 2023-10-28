use cgmath::{Vector2, Point3, Point2, point3};

use crate::{
    error::TError,
    model,
    resources::{load_glb_model, load_sprite},
    state::{GpuState, TeState},
};

use super::{
    model::InstancedModel, sprite::InstancedSprite, InstanceMap, InstanceReference,
    InstanceType, InstancedDraw,
};

enum ModelType {
    Normal(model::Model),
    Animated(model::AnimatedModel),
    None,
}

pub struct ModelBuilder<'state, 'gpu, 'a> {
    te_state: &'state mut TeState,
    model_name: &'a str,
    gpu: &'gpu GpuState,
    position: Point3<f32>,
    absolute_position: bool,
    model: ModelType,
}

impl<'state, 'gpu, 'a> ModelBuilder<'state, 'gpu, 'a> {
    pub(crate) fn new(
        te_state: &'state mut TeState,
        model_name: &'a str,
        gpu: &'gpu GpuState,
        position: Point3<f32>,
    ) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder {
            te_state,
            model_name,
            gpu,
            position,
            absolute_position: false,
            model: ModelType::None,
        }
    }

    pub fn absolute_position(self) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder {
            absolute_position: true,
            ..self
        }
    }

    pub fn with_absolute_position(self, absolute_position: bool) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder {
            absolute_position,
            ..self
        }
    }

    pub fn with_model(self, model: model::Model) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder {
            model: ModelType::Normal(model),
            ..self
        }
    }

    pub fn with_animated_model(
        self,
        model: model::AnimatedModel,
    ) -> ModelBuilder<'state, 'gpu, 'a> {
        ModelBuilder {
            model: ModelType::Animated(model),
            ..self
        }
    }

    pub fn build(self) -> Result<InstanceReference, TError> {
        let position = if !self.absolute_position {
            let tile_size = self.te_state.instances.tile_size;
            let x = self.position.x * tile_size.x;
            let y = self.position.y * tile_size.y;
            let z = self.position.z * tile_size.z;

            point3(x, y, z)
        } else {
            self.position
        };

        // TODO: Remove duplicate code inside this match statement
        match self.model {
            ModelType::Normal(model) => {
                let instances = &mut self.te_state.instances;

                if let Some(instanced_m) = instances.opaque_instances.instanced.get_mut(self.model_name) {
                    instanced_m.add_instance(position, &self.gpu.device)
                } else {
                    let model = Some(model).ok_or(TError::UninitializedModel)?;
                    let transparent_meshes = model.transparent_meshes.len();
                    let instanced_m = InstancedModel::new(model, &self.gpu.device, position);
                    instances
                        .opaque_instances.instanced
                        .insert(self.model_name.to_string(), instanced_m);
                    if transparent_meshes > 0 {
                        instances
                            .transparent_instances
                            .insert(self.model_name.to_string());
                    }
                }

                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D,
                };

                reference.index = instances
                    .opaque_instances.instanced
                    .instance(&reference)
                    .instances
                    .len()
                    - 1;

                instances.render_matrix.register_instance(
                    reference.clone(),
                    position,
                    instances
                        .opaque_instances
                        .instanced
                        .instance(&reference)
                        .model
                        .get_extremes(),
                );

                Ok(reference)
            }
            ModelType::Animated(mut model) => {
                let instances = &mut self.te_state.instances;

                model.set_instance_position(0, position, &self.gpu.queue); // TODO: Investigate why this is needed
                let transparent_meshes = model.transparent_meshes.len();
                let position = model.instance.position;
                instances
                    .opaque_instances
                    .animated
                    .insert(self.model_name.to_string(), model);
                if transparent_meshes > 0 {
                    instances
                        .transparent_instances
                        .insert(self.model_name.to_string());
                }

                let reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // TODO: Should index be 0?
                    dimension: InstanceType::Anim3D,
                };

                instances.render_matrix.register_instance(
                    reference.clone(),
                    position,
                    instances
                        .opaque_instances
                        .animated
                        .instance(&reference)
                        .get_extremes(),
                );

                Ok(reference)
            }
            ModelType::None => {
                let instances = &mut self.te_state.instances;

                if let Some(instanced_m) = instances.opaque_instances.instanced.get_mut(self.model_name) {
                    instanced_m.add_instance(position, &self.gpu.device);
                } else {
                    let model = load_glb_model(
                        self.model_name,
                        &self.gpu.device,
                        &self.gpu.queue,
                        &instances.layout,
                        instances.resources_path.clone(),
                        &instances.default_texture_path,
                    )
                    .map_err(|_| TError::GLBModelLoadingFail)?;
                    let transparent_meshes = model.transparent_meshes.len();
                    let instanced_m = InstancedModel::new(model, &self.gpu.device, position);
                    instances
                        .opaque_instances
                        .instanced
                        .insert(self.model_name.to_string(), instanced_m);
                    if transparent_meshes > 0 {
                        instances
                            .transparent_instances
                            .insert(self.model_name.to_string());
                    }
                }

                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D,
                };

                reference.index = instances
                    .opaque_instances
                    .instanced
                    .instance(&reference)
                    .instances
                    .len()
                    - 1;

                instances.render_matrix.register_instance(
                    reference.clone(),
                    position,
                    instances
                        .opaque_instances
                        .instanced
                        .instance(&reference)
                        .model
                        .get_extremes(),
                );

                Ok(reference)
            }
        }
    }
}

pub struct SpriteBuilder<'state, 'gpu, 'a, 'b> {
    te_state: &'state mut TeState,
    sprite_name: &'a str,
    gpu: &'gpu GpuState,
    position: Point2<f32>,
    depth: f32,
    force_new_instance_id: Option<&'b str>,
    size: Option<Vector2<f32>>,
    material: Option<model::Material>,
}

impl<'state, 'gpu, 'a, 'b> SpriteBuilder<'state, 'gpu, 'a, 'b> {
    pub(crate) fn new(
        te_state: &'state mut TeState,
        sprite_name: &'a str,
        gpu: &'gpu GpuState,
        position: Point2<f32>,
        depth: f32
    ) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
        SpriteBuilder {
            te_state,
            sprite_name,
            gpu,
            position,
            depth,
            force_new_instance_id: None,
            size: None,
            material: None,
        }
    }

    pub fn with_id(self, id: &'b str) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
        SpriteBuilder {
            force_new_instance_id: Some(id),
            ..self
        }
    }

    pub fn with_size(self, size: Vector2<f32>) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
        SpriteBuilder {
            size: Some(size),
            ..self
        }
    }

    pub fn with_material(self, material: model::Material) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
        SpriteBuilder {
            material: Some(material),
            ..self
        }
    }

    pub fn build(self) -> Result<InstanceReference, TError> {
        // TODO: refractor
        if let Some(material) = self.material {
            let instances = &mut self.te_state.instances;
            let screen_size = Vector2::new(
                self.te_state.size.width,
                self.te_state.size.height
            );

            let instance_name =
                self.sprite_name.to_string() + self.force_new_instance_id.unwrap_or_default();
            if let Some(instanced_s) = instances.sprite_instances.instanced.get_mut(&instance_name) {
                instanced_s.add_instance(
                    self.position,
                    self.size,
                    &self.gpu.device,
                    screen_size
                );
            } else {
                let instanced_s = InstancedSprite::new(
                    material,
                    &self.gpu.device,
                    self.position,
                    self.depth,
                    self.size.ok_or(TError::SizeRequired)?,
                    screen_size
                );
                instances
                    .sprite_instances
                    .instanced
                    .insert(instance_name.to_string(), instanced_s);
            }

            let mut instance_ref = InstanceReference {
                name: instance_name.to_string(),
                index: 0, // 0 is placeholder
                dimension: InstanceType::Sprite,
            };

            instance_ref.index = instances
                .sprite_instances
                .instanced
                .instance(&instance_ref)
                .instances
                .len()
                - 1;

            Ok(instance_ref)
        } else {
            let instances = &mut self.te_state.instances;
            let screen_size = Vector2::new(
                self.te_state.size.width,
                self.te_state.size.height
            );

            let instance_name =
                self.sprite_name.to_string() + self.force_new_instance_id.unwrap_or_default();
            if let Some(instanced_s) = instances.sprite_instances.instanced.get_mut(&instance_name) {
                instanced_s.add_instance(
                    self.position,
                    self.size,
                    &self.gpu.device,
                    screen_size
                );
            } else {
                let (sprite, sprite_size) = load_sprite(
                    self.sprite_name,
                    &self.gpu.device,
                    &self.gpu.queue,
                    &instances.layout,
                    instances.resources_path.clone(),
                )
                .map_err(|_| TError::SpriteLoadingFail)?;
                let size = match self.size {
                    Some(size) => size,
                    None => sprite_size,
                };
                let instanced_s = InstancedSprite::new(
                    sprite,
                    &self.gpu.device,
                    self.position,
                    self.depth,
                    size,
                    screen_size
                );
                instances
                    .sprite_instances
                    .instanced
                    .insert(instance_name.clone(), instanced_s);
            }

            let mut inst_ref = InstanceReference {
                name: instance_name,
                index: 0, // 0 is placeholder
                dimension: InstanceType::Sprite,
            };

            inst_ref.index = instances
                .sprite_instances
                .instanced
                .instance(&inst_ref)
                .instances
                .len()
                - 1;

            Ok(inst_ref)
        }
    }
}
