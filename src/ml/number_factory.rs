use crate::ml::{
    NumberLike,
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
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

    fn activate_neuron(&mut self, neuron: N, activation: &NeuronActivation) -> N;
    fn activate_layer(&mut self, layer: &Vec<N>, activation: &LayerActivation) -> Vec<N>;
    fn compute_error(&mut self, expected: &Vec<N>, actual: &Vec<N>, error_function: &ErrorFunction) -> N;
}
