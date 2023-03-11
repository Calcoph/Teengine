struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
}

// Vertex shader

struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct PushConstants {
    index: u32
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
var<push_constant> push_constants: PushConstants;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) inst_index: u32
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) inst_index: u32
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    let model_matrix = mat4x4<f32> (
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3
    );
    let world_position = model_matrix * vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.inst_index = model.inst_index + push_constants.index;
    return out;
}

// Fragment shaders

@fragment
fn fs_main(
    in: VertexOutput,
) -> @location(0) u32 {
    return in.inst_index;
}

@fragment
fn fs_color(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    return vec4(f32((in.inst_index >> 16u) & 0xFFu)/255.0, f32((in.inst_index >> 8u) & 0xFFu)/255.0, f32(in.inst_index & 0xFFu)/255.0, 1.0);
}

