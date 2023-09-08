use cgmath::Vector3;

use crate::{state::{GpuState, TeState}, model, error::TError, resources::load_glb_model};

use super::{InstanceReference, model::InstancedModel, DrawModel, InstanceType, InstanceMap, InstancedDraw};

enum ModelType {
    Normal(model::Model),
    Animated(model::AnimatedModel),
    None
}

pub struct ModelBuilder<'state, 'gpu, 'a> {
    te_state: &'state mut TeState,
    model_name: &'a str,
    gpu: &'gpu GpuState,
    position: (f32, f32, f32),
    absolute_position: bool,
    model: ModelType
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

    pub fn with_animated_model(self, model: model::AnimatedModel) -> ModelBuilder<'state, 'gpu, 'a> {
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
                    instances.opaque_instances
                        .insert(self.model_name.to_string(), DrawModel::M(instanced_m));
                    if transparent_meshes > 0 {
                        instances.transparent_instances.insert(self.model_name.to_string());
                    }
                }
        
                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D,
                };
        
                reference.index = instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .instances
                    .len()
                    - 1;
        
                instances.render_matrix.register_instance(
                    reference.clone(),
                    cgmath::vec3(x, y, z),
                    instances.opaque_instances
                        .instance(&reference)
                        .get_m()
                        .model
                        .get_extremes(),
                );
        
                Ok(reference)
            },
            ModelType::Animated(mut model) => {
                let instances = &mut self.te_state.instances;

                model.set_instance_position(0, Vector3::new(x, y, z), &self.gpu.queue); // TODO: Investigate why this is needed
                let transparent_meshes = model.transparent_meshes.len();
                let position = model.instance.position;
                instances.opaque_instances
                    .insert(self.model_name.to_string(), DrawModel::A(model));
                if transparent_meshes > 0 {
                    instances.transparent_instances.insert(self.model_name.to_string());
                }

                let reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // TODO: Should index be 0?
                    dimension: InstanceType::Opaque3D,
                };

                instances.render_matrix.register_instance(
                    reference.clone(),
                    position,
                    instances.opaque_instances
                        .instance(&reference)
                        .get_a()
                        .get_extremes(),
                );

                Ok(reference)
            },
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
                    instances.opaque_instances
                        .insert(self.model_name.to_string(), DrawModel::M(instanced_m));
                    if transparent_meshes > 0 {
                        instances.transparent_instances.insert(self.model_name.to_string());
                    }
                }
        
                let mut reference = InstanceReference {
                    name: self.model_name.to_string(),
                    index: 0, // 0 is placeholder
                    dimension: InstanceType::Opaque3D,
                };
        
                reference.index = instances
                    .opaque_instances
                    .instance(&reference)
                    .get_m()
                    .instances
                    .len()
                    - 1;
        
                instances.render_matrix.register_instance(
                    reference.clone(),
                    cgmath::vec3(x, y, z),
                    instances.opaque_instances
                        .instance(&reference)
                        .get_m()
                        .model
                        .get_extremes(),
                );
        
                Ok(reference)
            },
        }
    }
}
