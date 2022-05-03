use super::math::{
    NumberLike,
    NumberFactory,
};

struct FloatFactory {}
impl NumberFactory<f32> for FloatFactory {
    fn zero() -> f32 {
        0.0
    }

    fn one() -> f32 {
        1.0
    }
}

impl NumberLike<FloatFactory> for f32 {
    fn exp(&self) -> f32 {
        self.exp()
    }

    fn powi(&self, power: i32) -> f32 {
        self.powi(power)
    }

    fn powf(&self, power: f32) -> f32 {
        self.powf(power)
    }

    fn sin(&self) -> f32 {
        self.sin()
    }

    fn cos(&self) -> f32 {
        self.cos()
    }

    fn tan(&self) -> f32 {
        self.tan()
    }

    fn asin(&self) -> f32 {
        self.asin()
    }

    fn acos(&self) -> f32 {
        self.acos()
    }

    fn atan(&self) -> f32 {
        self.atan()
    }

    fn sinh(&self) -> f32 {
        self.sinh()
    }

    fn cosh(&self) -> f32 {
        self.cosh()
    }

    fn tanh(&self) -> f32 {
        self.tanh()
    }

    fn asinh(&self) -> f32 {
        self.asinh()
    }

    fn acosh(&self) -> f32 {
        self.acosh()
    }

    fn atanh(&self) -> f32 {
        self.atanh()
    }

    fn log(&self, base: f32) -> f32 {
        self.log(base)
    }

    fn sqrt(&self) -> f32 {
        self.sqrt()
    }

    fn cbrt(&self) -> f32 {
        self.cbrt()
    }

    fn relu(&self) -> f32 {
        self.relu()
    }

    fn leaky_relu(&self, leaking_factor: f32) -> f32 {
        if *self > 0.0 {
            1.0
        } else {
            leaking_factor
        }
    }

    fn abs(&self) -> f32 {
        self.abs()
    }
}
