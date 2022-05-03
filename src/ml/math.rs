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
}

pub trait NumberLike<Factory> where
    Self:
        Sized + Default + Clone +
        cmp::PartialEq + cmp::PartialOrd +
        ops::Add<Output=Self> + ops::AddAssign +
        ops::Sub<Output=Self> + ops::SubAssign +
        ops::Neg<Output=Self> +
        ops::Mul<Output=Self> + ops::MulAssign +
        ops::Div<Output=Self> + ops::DivAssign,
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
pub enum SingleActivation {
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

pub trait SingleActivator<T: NumberLike<F>, F: NumberFactory<T>> {
    fn activate(&self, x: &T) -> T;
}

pub trait LayerActivator<T: NumberLike<F>, F: NumberFactory<T>> {
    fn activate(&self, x: &[T]) -> Vec<T>;
}

pub trait ErrorComputer<T: NumberLike<F>, F: NumberFactory<T>> {
    fn compute_error(&self, x: &[T], y: &[T]) -> T;
}
