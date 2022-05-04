mod traits;

use std::{
    ops,
};

use rand::prelude::*;

use crate::ml::{
    ad::{AD},
    math::{
        NumberLike,
        NumberFactory,
        CellActivator,
        CellActivation,
        LayerActivator,
        LayerActivation,
        ErrorComputer,
        ErrorFunction,
    },
};

#[derive(Copy, Clone)]
pub struct ADNumber<'a> {
    ad: Option<&'a AD>,
    id: usize,
    tape_id: usize,
    scalar: f32,
}

pub struct ADNumberFactory {
    ad: AD,
    rng: ThreadRng,
}

impl <'a> ADNumberFactory {
    fn get_instance() -> &'a ADNumberFactory {
        unsafe {
            match AD_NUMBER_FACTORY {
                Some(ref factory) => factory,
                None => {
                    let factory = ADNumberFactory {
                        ad: AD::new(),
                        rng: thread_rng(),
                    };
                    AD_NUMBER_FACTORY = Some(factory);
                    AD_NUMBER_FACTORY.as_ref().unwrap()
                }
            }
        }
    }

    fn get_instance_mut() -> &'a mut ADNumberFactory {
        unsafe {
            AD_NUMBER_FACTORY.get_or_insert(ADNumberFactory {
                ad: AD::new(),
                rng: thread_rng(),
            })
        }
    }
}

impl <'a> NumberFactory<ADNumber<'a>> for ADNumberFactory {
    fn zero() -> ADNumber<'a> {
        ADNumber::new(0, 0, None, 0.0)
    }

    fn one() -> ADNumber<'a> {
        ADNumber::new(0, 0, None, 1.0)
    }

    fn small_rand() -> ADNumber<'a> {
        let factory = ADNumberFactory::get_instance_mut();
        let scalar = factory.rng.gen::<f32>();
        factory.ad.create_variable(scalar)
    }
}

static mut AD_NUMBER_FACTORY: Option<ADNumberFactory> = None;

impl <'a> ADNumber<'a> {
    pub fn new(
        tape_id: usize,
        id: usize,
        ad: Option<&'a AD>,
        scalar: f32,
    ) -> ADNumber<'a> {
        ADNumber {
            tape_id,
            id,
            ad,
            scalar,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn tape_id(&self) -> usize {
        self.tape_id
    }

    pub fn ad(&self) -> &AD {
        self.ad
            .expect(
                "ADNumber should have an AD instance reference"
            )
    }

    pub fn scalar(&self) -> f32 {
        self.scalar
    }

    // This is not differentiable
    pub fn is_constant(&self) -> bool {
        self.tape_id == 0
    }

    // This is differentiable
    pub fn is_variable(&self) -> bool {
        self.tape_id > 0
    }

    pub fn diff(&self, wrt: &ADNumber) -> f32 {
        self.ad.expect("missing AD instance reference")
            .diff(self, wrt)
    }
}

impl <'a> NumberLike<ADNumberFactory> for ADNumber<'a> {
    fn exp(&self) -> ADNumber<'a> {
        let exp = self.scalar.exp();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, exp, exp
            )
    }

    fn log(&self, base: f32) -> Self {
        let log = self.scalar.log(base);
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, log, 1.0 / self.scalar
            )
    }

    fn powi(&self, operand: i32) -> Self {
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, self.scalar.powi(operand),
                self.scalar.powi(operand - 1) * operand as f32
            )
    }

    fn powf(&self, operand: f32) -> Self {
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, self.scalar.powf(operand),
                self.scalar.powf(operand - 1.0) * operand
            )
    }

    fn sin(&self) -> Self {
        let sin = self.scalar.sin();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, sin, self.scalar.cos()
            )
    }

    fn cos(&self) -> Self {
        let cos = self.scalar.cos();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, cos, -self.scalar.sin()
            )
    }

    fn tan(&self) -> Self {
        let tan = self.scalar.tan();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, tan, 1.0 + tan * tan
            )
    }

    fn asin(&self) -> Self {
        let asin = self.scalar.asin();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, asin, 1.0 / (1.0 - self.scalar.powi(2)).sqrt()
            )
    }

    fn acos(&self) -> Self {
        let acos = self.scalar.acos();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, acos, -1.0 / (1.0 - self.scalar.powi(2)).sqrt(),
            )
    }

    fn atan(&self) -> Self {
        let atan = self.scalar.atan();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, atan, 1.0 / (1.0 + self.scalar.powi(2)),
            )
    }

    fn sinh(&self) -> Self {
        let sinh = self.scalar.sinh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, sinh, self.scalar.cosh(),
            )
    }

    fn cosh(&self) -> Self {
        let cosh = self.scalar.cosh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, cosh, self.scalar.sinh(),
            )
    }

    fn tanh(&self) -> Self {
        let tanh = self.scalar.tanh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, tanh, 1.0 - tanh * tanh,
            )
    }

    fn asinh(&self) -> Self {
        let asinh = self.scalar.asinh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, asinh, 1.0 / (self.scalar.powi(3)).sqrt(),
            )
    }

    fn acosh(&self) -> Self {
        let acosh = self.scalar.acosh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, acosh, 1.0 / (self.scalar.powi(2) - 1.0).sqrt(),
            )
    }

    fn atanh(&self) -> Self {
        let atanh = self.scalar.atanh();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, atanh, 1.0 / (1.0 - self.scalar.powi(2)),
            )
    }

    fn sqrt(&self) -> Self {
        let sqrt = self.scalar.sqrt();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, sqrt, 0.5 / sqrt,
            )
    }

    fn cbrt(&self) -> Self {
        let cbrt = self.scalar.cbrt();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, cbrt,
                1.0 / 3.0 * (self.scalar / cbrt).powi(2)
            )
    }

    fn abs(&self) -> Self {
        let abs = self.scalar.abs();
        self.ad.expect("missing AD instance reference")
            .create_unary_composite(
                self, abs,
                if self.scalar < 0.0 { -1.0 } else { 1.0 }
            )
    }
}

impl <'a> ops::Add<ADNumber<'a>> for ADNumber<'a> {
    type Output  = ADNumber<'a>;

    fn add(self, other: ADNumber<'a>) -> Self::Output {
        self.ad.expect("reference to AD instance is missing")
            .create_binary_composite(
                &self, &other,
                self.scalar + other.scalar,
                1.0, 1.0,
            )
    }
}

impl <'a> ops::Add<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn add(self, other: f32) -> ADNumber<'a> {
        self.ad.expect("reference to AD instance is missing")
            .create_unary_composite(
                &self, self.scalar + other, 1.0
            )
    }
}

impl <'a> ops::AddAssign<ADNumber<'a>> for ADNumber<'a> {
    fn add_assign(&mut self, other: ADNumber<'a>) {
        *self = *self + other;
    }
}

impl <'a> ops::Sub<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn sub(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.expect("reference to AD instance is missing")
            .create_binary_composite(
                &self, &other,
                self.scalar - other.scalar,
                1.0, -1.0,
            )
    }
}

impl <'a> ops::Sub<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn sub(self, other: f32) -> ADNumber<'a> {
        self.ad.expect("reference to AD instance is missing")
            .create_unary_composite(
                &self, self.scalar - other, 1.0
            )
    }
}

impl <'a> ops::SubAssign<ADNumber<'a>> for ADNumber<'a> {
    fn sub_assign(&mut self, other: ADNumber<'a>) {
        *self = *self + other;
    }
}

impl <'a> ops::Neg for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn neg(self) -> ADNumber<'a> {
        self.ad.expect("reference to AD instance is missing")
            .create_unary_composite(
                &self, -self.scalar, -1.0
            )
    }
}

impl <'a> ops::Mul<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn mul(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.expect("reference to AD instance is missing")
            .create_binary_composite(
                &self, &other,
                self.scalar * other.scalar,
                other.scalar, self.scalar,
            )
    }
}

impl <'a> ops::Mul<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn mul(self, other: f32) -> ADNumber<'a> {
        self.ad.expect("reference to AD instance is missing")
            .create_unary_composite(
                &self, self.scalar * other, other
            )
    }
}

impl <'a> ops::MulAssign<ADNumber<'a>> for ADNumber<'a> {
    fn mul_assign(&mut self, other: ADNumber<'a>) {
        *self = *self * other;
    }
}

impl <'a> ops::Div<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn div(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.expect("reference to AD instance is missing")
            .create_binary_composite(
                &self, &other,
                self.scalar / other.scalar,
                1.0 / other.scalar, -self.scalar / other.scalar.powi(2),
            )
    }
}

impl <'a> ops::Div<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn div(self, other: f32) -> ADNumber<'a> {
        self.ad.expect("reference to AD instance is missing")
            .create_unary_composite(
                &self, self.scalar / other, 1.0 / other
            )
    }
}

impl <'a> ops::DivAssign<ADNumber<'a>> for ADNumber<'a> {
    fn div_assign(&mut self, other: ADNumber<'a>) {
        *self = *self / other;
    }
}

impl <'a> CellActivator<ADNumber<'a>, ADNumberFactory> for CellActivation {
    fn activate(&self, value: &ADNumber<'a>) -> ADNumber<'a> {
        match *self {
            CellActivation::None => value.clone(),
            CellActivation::ReLu => value.relu(),
            CellActivation::LeakyReLU(alpha) => value.leaky_relu(
                ADNumberFactory::one() * alpha
            ),
        }
    }
}

impl <'a> LayerActivator<ADNumber<'a>, ADNumberFactory> for LayerActivation {
    fn activate(&self, value: &[ADNumber<'a>]) -> Vec<ADNumber<'a>> {
        match *self {
            LayerActivation::None => value.to_vec(),
            LayerActivation::SoftMax => {
                let mut sum = ADNumberFactory::zero();
                let mut result = value.to_vec();

                for (i, v) in value.iter().enumerate() {
                    let exp = v.exp();
                    sum = sum + exp;
                    result[i] = exp;
                }

                result.iter().map(|v| *v / sum).collect::<Vec<_>>()
            },
        }
    }
}

impl <'a> ErrorComputer<ADNumber<'a>, ADNumberFactory> for ErrorFunction {
    fn compute_error(&self, value: &[ADNumber<'a>], target: &[ADNumber<'a>]) -> ADNumber<'a> {
        match *self {
            ErrorFunction::None => ADNumberFactory::one(),
            ErrorFunction::EuclideanDistanceSquared => {
                let mut sum = ADNumberFactory::zero();

                for (v, t) in value.iter().zip(target.iter()) {
                    sum += (*v - *t).powi(2);
                }

                sum
            },
        }
    }
}
