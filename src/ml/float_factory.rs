use crate::ml::{
    NumberLike,
    NumberFactory,
};

pub struct FloatFactory {

}

impl FloatFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl NumberLike for f32 {

}

impl NumberFactory<f32> for FloatFactory {
    fn create_variable(&mut self, scalar: f32) -> f32 {
        scalar
    }

    fn create_random_variable(&mut self) -> f32 {
        rand::random::<f32>()
    }

    fn multiply(&mut self, left: f32, right: f32) -> f32 {
        left * right
    }

    fn divide(&mut self, left: f32, right: f32) -> f32 {
        left / right
    }

    fn addition(&mut self, left: f32, right: f32) -> f32 {
        left + right
    }

    fn subtract(&mut self, left: f32, right: f32) -> f32 {
        left - right
    }

    fn exp(&mut self, operand: f32) -> f32 {
        operand.exp()
    }
}
