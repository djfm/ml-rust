use std::{
    cmp, ops,
};

pub trait NumberFactory<Number>
    where
        Self: Sized,
        Number: NumberLike<Self>
{
    fn zero() -> Number;
    fn one() -> Number;
    fn small_rand() -> Number;
    fn small_random() -> Number {
        Self::small_rand()
    }
    fn from_scalar(scalar: f32) -> Number;
}

pub trait NumberLike<Factory>:
    Sized + Default + Clone + Copy +
    cmp::PartialEq + cmp::PartialOrd +
    ops::Add<Output=Self> + ops::AddAssign +
    ops::Sub<Output=Self> + ops::SubAssign +
    ops::Neg<Output=Self> +
    ops::Mul<Output=Self> + ops::MulAssign +
    ops::Div<Output=Self> + ops::DivAssign
where
    Factory: NumberFactory<Self>
{
    fn relu(&self) -> Self {
        if *self < Factory::zero() {
            Factory::zero()
        } else {
            self.clone()
        }
    }

    fn leaky_relu(&self, leaking_factor: Self) -> Self {
        if *self < Factory::zero() {
            Factory::zero()
        } else {
            leaking_factor
        }
    }

    fn sigmoid(&self) -> Self {
        Factory::one() / (Factory::one() + -self.exp())
    }

    fn exp(&self) -> Self;
    fn powi(&self, power: i32) -> Self;
    fn powf(&self, power: f32) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn log(&self, base: f32) -> Self;
    fn ln(&self) -> Self {
        self.log(1.0f32.exp())
    }
    fn log10(&self) -> Self {
        self.log(10.0f32)
    }
    fn sqrt(&self) -> Self;
    fn cbrt(&self) -> Self;
    fn abs(&self) -> Self;

    fn neg(&self) -> Self {
        Factory::zero() - self.clone()
    }
}

pub trait Differentiable<N: NumberLike<F>, F: NumberFactory<N>> {
    fn diff(&self, wrt: &N) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellActivation {
    None,
    ReLu,
    LeakyReLU(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerActivation {
    None,
    SoftMax,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorFunction {
    None,
    EuclideanDistanceSquared,
}

impl CellActivation {
    pub fn compute<N: NumberLike<F>, F: NumberFactory<N>>(
        &self,
        x: &N
    ) -> N {
        match self {
            CellActivation::None => x.clone(),
            CellActivation::ReLu => x.relu(),
            CellActivation::LeakyReLU(leaking_factor) => x.leaky_relu(
                NumberFactory::from_scalar(*leaking_factor)
            ),
        }
    }
}

impl LayerActivation {
    pub fn compute<N: NumberLike<F>, F: NumberFactory<N>>(
        &self,
        x: &Vec<N>
    ) -> Vec<N> {
        match self {
            LayerActivation::None => x.clone(),
            LayerActivation::SoftMax => {
                let mut sum = F::zero();
                let mut res = vec![F::zero(); x.len()];

                for (i, v) in x.iter().enumerate() {
                    let exp = v.exp();
                    sum += exp;
                    res[i] = exp;
                }

                for v in res.iter_mut() {
                    *v /= sum;
                }

                res
            }
        }
    }
}

impl ErrorFunction {
    pub fn compute<N: NumberLike<F>, F: NumberFactory<N>>(
        &self,
        expected: &Vec<N>,
        actual: &Vec<N>,
    ) -> N {
        match self {
            ErrorFunction::None => panic!("No error function specified for neural network"),
            ErrorFunction::EuclideanDistanceSquared => {
                let mut sum = F::zero();

                for (a, e) in actual.iter().zip(expected.iter()) {
                    let diff = *a - *e;
                    sum += diff.powi(2);
                }

                sum
            }
        }
    }
}

pub fn one_hot_label<N: NumberLike<F>, F: NumberFactory<N>>(vector: &Vec<N>) -> usize {
    let mut max = vector[0];
    let mut max_index = 0;

    for (i, v) in vector.iter().enumerate() {
        if *v > max {
            max = *v;
            max_index = i;
        }
    }

    max_index
}
