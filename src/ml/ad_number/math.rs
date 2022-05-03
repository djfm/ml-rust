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
}

pub trait NumberLike<Factory> where
    Self:
        Sized + Default +
        cmp::PartialEq + cmp::PartialOrd +
        ops::Add + ops::AddAssign +
        ops::Sub + ops::SubAssign,
    Factory: NumberFactory<Self>
{
    fn relu(self) -> Self {
        if self < Factory::zero() {
            Factory::zero()
        } else {
            self
        }
    }

    fn leaky_relu(self, leaking_factor: Self) -> Self {
        if self < Factory::zero() {
            Factory::zero()
        } else {
            leaking_factor
        }
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
}
