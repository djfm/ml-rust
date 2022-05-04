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

#[derive(Debug, Clone, Copy)]
pub enum CellActivation {
    None,
    ReLu,
    LeakyReLU(f32),
}

#[derive(Clone, Copy, Debug)]
pub enum LayerActivation {
    None,
    SoftMax,
}

#[derive(Clone, Copy, Debug)]
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
