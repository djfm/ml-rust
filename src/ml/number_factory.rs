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
    fn get_as_differentiable(&mut self) -> Option<&mut (dyn DifferentiableNumberFactory<N>)>;

    fn from_scalar(&mut self, scalar: f32) -> N;
    fn from_scalars(&mut self, scalars: &[f32]) -> Vec<N> {
        scalars.iter().map(|&s| self.from_scalar(s)).collect()
    }

    fn hottest_index(&self, activations: &[N]) -> usize {
        if activations.is_empty() {
            panic!("activations is empty");
        }

        let mut max_index = 0;
        let mut max_activation = activations[0].scalar();

        for (i, activation) in activations.iter().enumerate() {
            if activation.scalar() > max_activation {
                max_index = i;
                max_activation = activation.scalar();
            }
        }

        max_index
    }

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
}

pub trait DifferentiableNumberFactory<N>: NumberFactory<N> where N: NumberLike {
    fn diff(&mut self, y: &N, x: &N) -> f32;
}

pub enum NumberFactoryWrapper<'a, N, F, D> where
    N: NumberLike,
    F: NumberFactory<N>,
    D: DifferentiableNumberFactory<N>
{
    Regular(&'a mut F),
    Differentiable(&'a mut D),
    _None(std::marker::PhantomData<N>),
}
