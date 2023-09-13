use cgmath::Vector3;

use crate::{
    error::TError,
    model,
    resources::{load_glb_model, load_sprite},
    state::{GpuState, TeState},
};

use super::{
    model::InstancedModel, sprite::InstancedSprite, DrawModel, InstanceMap, InstanceReference,
    InstanceType, InstancedDraw, QuadTreeElement,
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
    position: (f32, f32, f32),
    absolute_position: bool,
    model: ModelType,
}

impl<'state, 'gpu, 'a> ModelBuilder<'state, 'gpu, 'a> {
    pub(crate) fn new(
        te_state: &'state mut TeState,
        model_name: &'a str,
        gpu: &'gpu GpuState,
        position: (f32, f32, f32),
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
        let (x, y, z) = if !self.absolute_position {
            let tile_size = self.te_state.instances.tile_size;
            let x = self.position.0 * tile_size.0;
            let y = self.position.1 * tile_size.1;
            let z = self.position.2 * tile_size.2;

            (x, y, z)
        } else {
            self.position
        };

        // TODO: Remove duplicate code inside this match statement
        match self.model {
            ModelType::Normal(model) => {
                let instances = &mut self.te_state.instances;

                if let Some(instanced_m) = instances.opaque_instances.get_mut(self.model_name) {
                    match instanced_m {
                        DrawModel::M(m) => m.add_instance(x, y, z, &self.gpu.device),
                        DrawModel::A(_) => unreachable!(),
                    }
                } else {
                    let model = Some(model).ok_or(TError::UninitializedModel)?;
                    let transparent_meshes = model.transparent_meshes.len();
                    let instanced_m = InstancedModel::new(model, &self.gpu.device, x, y, z);
                    instances
                        .opaque_instances
                        .insert(self.model_name.to_string(), DrawModel::M(instanced_m));
                    if transparent_meshes > 0 {
                        instances
                            .transparent_instances
                            .insert(self.model_name.to_string());
                    }
                }

                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D { id: None },
                };

                reference.index = instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .instances
                    .len()
                    - 1;

                let id = instances.qtree.insert(QuadTreeElement::new(reference.clone(), x, z, instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .model
                    .get_extremes()));

                reference.dimension = InstanceType::Opaque3D { id };

                Ok(reference)
            }
            ModelType::Animated(mut model) => {
                let instances = &mut self.te_state.instances;

                model.set_instance_position(0, Vector3::new(x, y, z), &self.gpu.queue); // TODO: Investigate why this is needed
                let transparent_meshes = model.transparent_meshes.len();
                let position = model.instance.position;
                instances
                    .opaque_instances
                    .insert(self.model_name.to_string(), DrawModel::A(model));
                if transparent_meshes > 0 {
                    instances
                        .transparent_instances
                        .insert(self.model_name.to_string());
                }

                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // TODO: Should index be 0?
                    dimension: InstanceType::Opaque3D { id: None },
                };

                let id = instances.qtree.insert(QuadTreeElement::new(reference.clone(), position.x, position.z, instances
                    .opaque_instances
                    .instance(&reference)
                    .get_a()
                    .get_extremes()));

                reference.dimension = InstanceType::Opaque3D { id };

                Ok(reference)
            }
            ModelType::None => {
                let instances = &mut self.te_state.instances;

                if let Some(instanced_m) = instances.opaque_instances.get_mut(self.model_name) {
                    let instanced_m = instanced_m.get_mut_m();
                    instanced_m.add_instance(x, y, z, &self.gpu.device);
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
                    let instanced_m = InstancedModel::new(model, &self.gpu.device, x, y, z);
                    instances
                        .opaque_instances
                        .insert(self.model_name.to_string(), DrawModel::M(instanced_m));
                    if transparent_meshes > 0 {
                        instances
                            .transparent_instances
                            .insert(self.model_name.to_string());
                    }
                }

                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D { id: None },
                };

                reference.index = instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .instances
                    .len()
                    - 1;

                let id = instances.qtree.insert(QuadTreeElement::new(reference.clone(), x, z, instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .model
                    .get_extremes()));

                reference.dimension = InstanceType::Opaque3D { id };

                Ok(reference)
            }
        }
    }
}

pub struct SpriteBuilder<'state, 'gpu, 'a, 'b> {
    te_state: &'state mut TeState,
    sprite_name: &'a str,
    gpu: &'gpu GpuState,
    position: (f32, f32, f32),
    force_new_instance_id: Option<&'b str>,
    size: Option<(f32, f32)>,
    material: Option<model::Material>,
}

impl<'state, 'gpu, 'a, 'b> SpriteBuilder<'state, 'gpu, 'a, 'b> {
    pub(crate) fn new(
        te_state: &'state mut TeState,
        sprite_name: &'a str,
        gpu: &'gpu GpuState,
        position: (f32, f32, f32),
    ) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
        SpriteBuilder {
            te_state,
            sprite_name,
            gpu,
            position,
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

    pub fn with_size(self, size: (f32, f32)) -> SpriteBuilder<'state, 'gpu, 'a, 'b> {
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
            let screen_w = self.te_state.size.width;
            let screen_h = self.te_state.size.height;

            let instance_name =
                self.sprite_name.to_string() + self.force_new_instance_id.unwrap_or_default();
            if let Some(instanced_s) = instances.sprite_instances.get_mut(&instance_name) {
                instanced_s.add_instance(
                    self.position.0,
                    self.position.1,
                    self.size,
                    &self.gpu.device,
                    screen_w,
                    screen_h,
                );
            } else {
                let (width, height) = match self.size {
                    Some((w, h)) => (w, h),
                    None => return Err(TError::SizeRequired),
                };
                let instanced_s = InstancedSprite::new(
                    material,
                    &self.gpu.device,
                    self.position.0,
                    self.position.1,
                    self.position.2,
                    width,
                    height,
                    screen_w,
                    screen_h,
                );
                instances
                    .sprite_instances
                    .insert(instance_name.to_string(), instanced_s);
            }

            let mut instance_ref = InstanceReference {
                name: instance_name.to_string(),
                index: 0, // 0 is placeholder
                dimension: InstanceType::Sprite,
            };

            instance_ref.index = instances
                .sprite_instances
                .instance(&instance_ref)
                .instances
                .len()
                - 1;

            Ok(instance_ref)
        } else {
            let instances = &mut self.te_state.instances;
            let screen_w = self.te_state.size.width;
            let screen_h = self.te_state.size.height;

            let instance_name =
                self.sprite_name.to_string() + self.force_new_instance_id.unwrap_or_default();
            if let Some(instanced_s) = instances.sprite_instances.get_mut(&instance_name) {
                instanced_s.add_instance(
                    self.position.0,
                    self.position.1,
                    self.size,
                    &self.gpu.device,
                    screen_w,
                    screen_h,
                );
            } else {
                let (sprite, width, height) = load_sprite(
                    self.sprite_name,
                    &self.gpu.device,
                    &self.gpu.queue,
                    &instances.layout,
                    instances.resources_path.clone(),
                )
                .map_err(|_| TError::SpriteLoadingFail)?;
                let (width, height) = match self.size {
                    Some((w, h)) => (w, h),
                    None => (width, height),
                };
                let instanced_s = InstancedSprite::new(
                    sprite,
                    &self.gpu.device,
                    self.position.0,
                    self.position.1,
                    self.position.2,
                    width,
                    height,
                    screen_w,
                    screen_h,
                );
                instances
                    .sprite_instances
                    .insert(instance_name.clone(), instanced_s);
            }

            let mut inst_ref = InstanceReference {
                name: instance_name,
                index: 0, // 0 is placeholder
                dimension: InstanceType::Sprite,
            };

            inst_ref.index = instances
                .sprite_instances
                .instance(&inst_ref)
                .instances
                .len()
                - 1;

            Ok(inst_ref)
        }
    }
}
