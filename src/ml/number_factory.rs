use crate::ml::{
    NumberLike,
};

pub trait NumberFactory<N> where N: NumberLike {
    fn create_variable(&mut self, scalar: f32) -> N;
    fn create_random_variable(&mut self) -> N;

    fn multiply(&mut self, left: N, right: N) -> N;
    fn divide(&mut self, left: N, right: N) -> N;
    fn addition(&mut self, left: N, right: N) -> N;
    fn subtract(&mut self, left: N, right: N) -> N;

    fn exp(&mut self, operand: N) -> N;

    fn diff(&mut self, y: N, x: N) -> f32;
}
