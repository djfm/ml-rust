use crate::ml::{
    NumberLike,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronActivation {
    None,
    ReLu,
    LeakyRelu(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerActivation {
    None,
    SoftMax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorFunction {
    None,
    EuclideanDistanceSquared,
    CategoricalCrossEntropy,
}

pub trait NumberFactory<N> where N: NumberLike {
    fn from_scalar(scalar: f32) -> N;

    fn add(&mut self, a: &N, b: &N) -> N;
    fn sub(&mut self, a: &N, b: &N) -> N;
    fn mul(&mut self, a: &N, b: &N) -> N;
    fn div(&mut self, a: &N, b: &N) -> N;
    fn exp(&mut self, a: &N) -> N;
    fn log(&mut self, a: &N) -> N;
    fn powi(&mut self, a: &N, i: i32) -> N;
    fn pow(&mut self, a: &N, b: &N) -> N;

    fn activate_neuron(&mut self, a: &N, activation: &NeuronActivation) -> N;
    fn activate_layer(&mut self, a: &[N], activation: &LayerActivation) -> Vec<N>;
    fn compute_error(&mut self, a: &[N], b: &[N], error_function: &ErrorFunction) -> N;

    fn diff(&mut self, y: &N, x: &N);
}
