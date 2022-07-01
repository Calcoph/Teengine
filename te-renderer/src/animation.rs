use std::time::Instant;

use std::time::Duration;

use cgmath::Vector3;

#[derive(Debug)]
pub struct Animation {
    translation: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
    start_time: Instant,
    duration: Duration,
    looping: bool,
}

impl Animation {
    pub fn new<
        T: Into<Vector3<f32>>,
        U: Into<Vector3<f32>>,
        V: Into<Vector3<f32>>,
    >(
        translation: T,
        rotation: U,
        scale: V,
        duration: Duration,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            translation,
            rotation,
            scale,
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_prepare<
        T: Into<Vector3<f32>>,
        U: Into<Vector3<f32>>,
        V: Into<Vector3<f32>>,
    >(
        translation: T,
        rotation: U,
        scale: V,
        duration: Duration,
        start_time: Instant,
        looping: bool,
    ) -> Self {
        Animation {
            translation: translation.into(),
            rotation: rotation.into(),
            scale: scale.into(),
            start_time,
            duration,
            looping,
        }
    }

    pub fn new_translation<T: Into<Vector3<f32>>>(
        translation: T,
        duration: Duration,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            translation,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_translation_prepare<T: Into<Vector3<f32>>>(
        translation: T,
        duration: Duration,
        start_time: Instant,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            translation,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_rotation<T: Into<Vector3<f32>>>(
        rotation: T,
        duration: Duration,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            (0.0, 0.0, 0.0),
            rotation,
            (0.0, 0.0, 0.0),
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_rotation_prepare<T: Into<Vector3<f32>>>(
        rotation: T,
        duration: Duration,
        start_time: Instant,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            (0.0, 0.0, 0.0),
            rotation,
            (0.0, 0.0, 0.0),
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_scale<T: Into<Vector3<f32>>>(
        scale: T,
        duration: Duration,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            scale,
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn new_scale_prepare<T: Into<Vector3<f32>>>(
        scale: T,
        duration: Duration,
        start_time: Instant,
        looping: bool,
    ) -> Self {
        Animation::new_prepare(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            scale,
            duration,
            Instant::now(),
            looping,
        )
    }

    pub fn get_translation(&mut self) -> Vector3<f32> {
        self.get_proportional(self.translation, Instant::now())
    }

    pub fn get_rotation(&mut self) -> Vector3<f32> {
        self.get_proportional(self.rotation, Instant::now())
    }

    pub fn get_scale(&mut self) -> Vector3<f32> {
        self.get_proportional(self.scale, Instant::now())
    }

    pub fn has_ended(&self) -> bool {
        (Instant::now() > self.start_time + self.duration) && !self.looping
    }

    fn get_proportional(&mut self, vec: Vector3<f32>, now: Instant) -> Vector3<f32> {
        if now > self.start_time + self.duration {
            if self.looping {
                self.start_time = self.start_time + self.duration;
                vec + self.get_proportional(vec, now)
            } else {
                vec
            }
        } else {
            let animated_proportion = (now - self.start_time).as_secs_f32() / self.duration.as_secs_f32();
            vector_proportion(vec, animated_proportion)
        }
    }
}

fn vector_proportion(vec: Vector3<f32>, mult: f32) -> Vector3<f32> {
    Vector3 {
        x: vec.x * mult,
        y: vec.y * mult,
        z: vec.z * mult
    }
}