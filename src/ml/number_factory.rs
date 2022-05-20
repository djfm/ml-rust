use crate::ml::{
    NumberLike,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuronActivation {
    None,
    ReLu,
    LeakyRelu(f32),
    Sigmoid,
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

pub trait PartialDiffsRecorderHelper<N> where N: NumberLike {
    fn log(dependent_variable: &N, partial_derivative: f32) -> Self;
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
    fn ln(&mut self, a: &N) -> N;
    fn powi(&mut self, a: &N, i: i32) -> N;
    fn pow(&mut self, a: &N, b: &N) -> N;
    fn neg(&mut self, a: &N) -> N;

    fn activate_neuron(&mut self, a: &N, activation: &NeuronActivation) -> N {
        let dnf = self.get_as_differentiable();

        match activation {
            NeuronActivation::None => a.clone(),

            NeuronActivation::ReLu => {
                if a.scalar() > 0.0 {
                    if let Some(dnf) = dnf {
                        dnf.compose(a.scalar(), vec![(&a, 1.0)])
                    } else {
                        a.clone()
                    }
                } else {
                    if let Some(dnf) = dnf {
                        dnf.compose(a.scalar(), vec![(&a, 0.0)])
                    } else {
                        self.from_scalar(0.0)
                    }
                }
            },

            NeuronActivation::LeakyRelu(leak) => {
                if a.scalar() > 0.0 {
                    if let Some(dnf) = dnf {
                        dnf.compose(a.scalar(), vec![(&a, 1.0)])
                    } else {
                        a.clone()
                    }
                } else {
                    if let Some(dnf) = dnf {
                        dnf.compose(0.0, vec![(&a, *leak)])
                    } else {
                        self.from_scalar(*leak)
                    }
                }
            },

            NeuronActivation::Sigmoid => {
                let dnf = self.get_as_differentiable();
                let res = 1.0 / (1.0 + (-a.scalar()).exp());

                if let Some(dnf) = dnf {
                    dnf.compose(res, vec![(&a, 1.0 * (1.0 - res))])
                } else {
                    self.from_scalar(res)
                }
            }
        }
    }

    fn activate_layer(&mut self, a: &[N], activation: &LayerActivation) -> Vec<N> {
        match activation {
            LayerActivation::None => a.to_vec(),

            LayerActivation::SoftMax => {
                let mut sum = self.from_scalar(0.0);
                let mut res = Vec::with_capacity(a.len());

                for v in a.iter() {
                    let exp = self.exp(v);
                    sum = self.add(&sum, &exp);
                    res.push(exp);
                }

                for v in res.iter_mut() {
                    *v = self.div(v, &sum);
                }

                res
            }
        }
    }

    fn compute_error(&mut self, expected: &[N], actual: &[N], error_function: &ErrorFunction) -> N {
        match error_function {
            ErrorFunction::None => self.from_scalar(0.0),

            ErrorFunction::EuclideanDistanceSquared => {
                let mut sum = self.from_scalar(0.0);
                for (a, b) in expected.iter().zip(actual.iter()) {
                    let diff = self.sub(a, b);
                    let square = self.powi(&diff, 2);
                    sum = self.add(&sum, &square);
                }
                sum
            },

            ErrorFunction::CategoricalCrossEntropy => {
                let mut sum = self.from_scalar(0.0);
                for (e, a) in expected.iter().zip(actual.iter()) {
                    let log = self.ln(&a);
                    let mul = self.mul(&log, &e);
                    sum = self.sub(&sum, &mul);
                }
                sum
            },
        }
    }
}

pub trait DifferentiableNumberFactory<N>: NumberFactory<N> where N: NumberLike {
    fn diff(&mut self, y: &N, x: &N) -> f32;
    fn compose(&mut self, result: f32, partials: Vec<(&N, f32)>) -> N;
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
