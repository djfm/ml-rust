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

impl NumberFactory<f32> for FloatFactory {
    fn has_automatic_diff(&self) -> bool {
        true
    }

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

    fn binary_operation(
        &mut self,
        _left: f32, _right: f32,
        result: f32,
        _diff_left: f32, _diff_right: f32,
    ) -> f32 {
        result
    }

    fn unary_operation(
        &mut self,
        _operand: f32,
        result: f32,
        _diff: f32,
    ) -> f32 {
        result
    }

    fn to_scalar(&self, number: f32) -> f32 {
        number
    }
}

impl NumberLike for f32 {}
