use crate::{
    NumberLike,
    NumberFactory,
    DifferentiableNumberFactory,
};

pub struct FloatFactory {}

impl FloatFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl NumberLike for f32 {
    fn scalar(&self) -> f32 {
        *self
    }
}

impl NumberFactory<f32> for FloatFactory {
    fn get_as_differentiable(&mut self) -> Option<&mut (dyn DifferentiableNumberFactory<f32>)> {
        None
    }

    fn constant(&mut self, scalar: f32) -> f32 {
        scalar
    }

    fn add(&mut self, a: f32, b: f32) -> f32 {
        a + b
    }

    fn sub(&mut self, a: f32, b: f32) -> f32 {
        a - b
    }

    fn mul(&mut self, a: f32, b: f32) -> f32 {
        a * b
    }

    fn div(&mut self, a: f32, b: f32) -> f32 {
        a / b
    }

    fn exp(&mut self, a: f32) -> f32 {
        a.exp()
    }

    fn ln(&mut self, a: f32) -> f32 {
        a.ln()
    }

    fn powi(&mut self, a: &f32, i: i32) -> f32 {
        a.powi(i)
    }

    fn pow(&mut self, a: f32, b: f32) -> f32 {
        a.powf(b)
    }

    fn neg(&mut self, a: &f32) -> f32 {
        -a
    }
}
