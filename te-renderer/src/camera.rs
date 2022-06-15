use cgmath::*;
use winit::{event::*, dpi};
use std::time::Duration;
use std::f32::consts::FRAC_PI_2;
use wgpu::util::DeviceExt;

use crate::initial_config::InitialConfiguration;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub const SAFE_CAMERA_ANGLE: f32 = FRAC_PI_2 - 0.0001;

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4]
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into()
        }
    }

    fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        // We're using Vector4 because of the uniforms 16 byte spacing requirement
        self.view_position = camera.get_homogeneous();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ProjectionUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    projection: [[f32; 4]; 4]
}

impl ProjectionUniform {
    fn new(matrix: Matrix4<f32>) -> Self {
        ProjectionUniform { projection: matrix.into() }
    }
}

pub struct CameraState {
    projection: Projection,
    projection_uniform: ProjectionUniform,
    projection_buffer: wgpu::Buffer,
    pub projection_bind_group: wgpu::BindGroup,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub (crate) camera_controller: CameraController,
}

impl CameraState {
    pub(crate) fn new(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        projection_bind_group_layout: &wgpu::BindGroupLayout,
        init_config: InitialConfiguration
    ) -> Self {
        let camera = Camera::new(
            init_config.camera_position,
            cgmath::Deg(init_config.camera_yaw),
            cgmath::Deg(init_config.camera_pitch)
        );
        let projection = Projection::new(
            config.width,
            config.height,
            init_config.camera_fovy,
            init_config.camera_znear,
            init_config.camera_zfar
        );
        let projection_uniform = ProjectionUniform::new(projection.calc_2d_matrix());
        let projection_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Projection Buffer"),
                contents: bytemuck::cast_slice(&[projection_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );
        let projection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &projection_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: projection_buffer.as_entire_binding()
                }
            ],
            label: Some("projeciton_bind_group")
        });
        let camera_controller = CameraController::new(
                init_config.camera_speed,
                init_config.camera_sensitivity
            );
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
            }
        );
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding()
                }
            ],
            label: Some("camera_bind_group")
        });

        CameraState {
            projection,
            projection_uniform,
            projection_buffer,
            projection_bind_group,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
        }
    }

    pub fn update(&mut self, dt: std::time::Duration, queue: &wgpu::Queue) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform.update_view_proj(&self.camera, &self.projection);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }

    pub fn resize(&mut self, new_size: dpi::PhysicalSize<u32>) {
        self.projection.resize(new_size.width, new_size.height);
    }

    pub fn get_fovy(&self) -> f32 {
        self.projection.fovy
    }

    pub fn get_znear(&self) -> f32 {
        self.projection.znear
    }
    
    pub fn get_zfar(&self) -> f32 {
        self.projection.zfar
    }

    pub fn get_speed(&self) -> f32 {
        self.camera_controller.speed
    }

    pub fn get_sensitivity(&self) -> f32 {
        self.camera_controller.sensitivity
    }

    pub fn get_yaw(&self) -> f32 {
        self.camera.yaw
    }

    pub fn get_pitch(&self) -> f32 {
        self.camera.pitch
    }

    pub fn set_fovy(&mut self, fovy: f32) {
        self.projection.fovy = fovy;
    }
    
    pub fn set_znear(&mut self, znear: f32) {
        self.projection.znear = znear;
    }

    pub fn set_zfar(&mut self, zfar: f32) {
        self.projection.zfar = zfar;
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.camera_controller.speed = speed;
    }

    pub fn set_sensitivity(&mut self, sensitivity: f32) {
        self.camera_controller.sensitivity = sensitivity
    }

    pub fn set_yaw(&mut self, yaw: f32) {
        self.camera.yaw = yaw
    }

    pub fn set_pitch(&mut self, pitch: f32) {
        self.camera.pitch = pitch
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.camera_controller.scroll = zoom
    }

    pub fn move_direction<V: Into<Vector3<f32>>>(&mut self, direction: V) {
        self.camera.position = self.camera.position + direction.into();
    }

    pub fn move_absolute<P: Into<Point3<f32>>>(&mut self, position: P) {
        self.camera.position = position.into();
    }

    pub fn get_position(&self) -> (f32, f32, f32) {
        return self.camera.position.into();
    }
}

#[derive(Debug)]
pub(crate) struct Camera {
    position: Point3<f32>,
    yaw: f32,
    pitch: f32
}

impl Camera {
    pub fn new<P: Into<Point3<f32>>, Y: Into<Rad<f32>>>(position: P, yaw: Y, pitch: Y) -> Self {
        Camera {
            position: position.into(),
            yaw: yaw.into().0,
            pitch: pitch.into().0
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_to_rh(
            self.position, Vector3::new(
                self.yaw.cos() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.sin() * self.pitch.cos()
            ).normalize(),
            Vector3::unit_y())
    }

    pub fn get_homogeneous(&self) -> [f32; 4]{
        self.position.to_homogeneous().into()
    }
}

pub struct Projection {
    width: u32,
    height: u32,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32
}

impl Projection {
    pub fn new(
        width: u32,
        height: u32,
        fovy: f32,
        znear: f32,
        zfar: f32
    ) -> Self {
        Self {
            width,
            height,
            aspect: width as f32 / height as f32,
            fovy: fovy,
            znear,
            zfar
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar)
    }

    pub fn calc_2d_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * ortho(0.0, self.width as f32, self.height as f32, 0.0, -1.0, 1.0)
    }
}

#[derive(Debug)]
pub(crate) struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    pub rotate_horizontal: f32,
    pub rotate_vertical: f32,
    pub scroll: f32,
    pub speed: f32,
    pub sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool{
        let amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Q => {
                self.rotate_horizontal -= amount/2.0;
                true
            }
            VirtualKeyCode::E => {
                self.rotate_horizontal += amount/2.0;
                true
            }
            VirtualKeyCode::X => {
                self.rotate_vertical -= amount/2.0;
                true
            }
            VirtualKeyCode::Z => {
                self.rotate_vertical += amount/2.0;
                true
            }
            VirtualKeyCode::R => {
                self.scroll += amount/2.0;
                true
            }
            VirtualKeyCode::F => {
                self.scroll -= amount/2.0;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw += self.rotate_horizontal * self.sensitivity * dt;
        camera.pitch += self.rotate_vertical * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -SAFE_CAMERA_ANGLE {
            camera.pitch = -SAFE_CAMERA_ANGLE;
        } else if camera.pitch > SAFE_CAMERA_ANGLE {
            camera.pitch = SAFE_CAMERA_ANGLE;
        }
    }
}