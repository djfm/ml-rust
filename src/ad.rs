use std::cell::RefCell;
use std::ops;
use std::collections::HashMap;

#[derive(Debug)]
struct PartialDerivative {
    with_respect_to_id: usize,
    differential: f32,
}

#[derive(Debug)]
struct TapeRecord {
    partials: Vec<PartialDerivative>
}

#[derive(Debug)]
struct Tape {
    records: Vec<TapeRecord>
}

type GradientsMap = HashMap<usize, Vec<f32>>;

pub trait NumberLike:
    ops::Add +
    ops::AddAssign +
    ops::Sub
    where
        Self: Sized
    {}


pub struct AD {
    tapes: RefCell<HashMap<usize, Tape>>,
    max_tape_id: RefCell<usize>,
    gradients: RefCell<GradientsMap>,
}

#[derive(Copy, Clone)]
pub struct ADNumber<'a> {
    ad: &'a AD,
    id: usize,
    tape_id: usize,
    scalar: f32,
}

impl Tape {
    fn new() -> Tape {
        Tape {
            records: Vec::new()
        }
    }
}

impl TapeRecord {
    fn new() -> TapeRecord {
        TapeRecord {
            partials: Vec::new()
        }
    }

    fn record(
        &mut self,
        with_respect_to: &ADNumber,
        differential: f32,
    ) -> &mut Self {
        self.partials.push(PartialDerivative {
            with_respect_to_id: with_respect_to.id,
            differential,
        });
        self
    }
}

impl <'a> ADNumber<'a> {
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
        self.ad.diff(self, wrt)
    }

    pub fn powi(&self, operand: i32) -> Self {
        self.ad.create_unary_composite(
            self, self.scalar.powi(operand),
            self.scalar.powi(operand - 1) * operand as f32,
        )
    }

    pub fn powf(&self, operand: f32) -> Self {
        self.ad.create_unary_composite(
            self, self.scalar.powf(operand),
            self.scalar.powf(operand - 1.0) * operand,
        )
    }

    pub fn exp(&self) -> Self {
        let exp = self.scalar.exp();
        self.ad.create_unary_composite(
            self, exp, exp,
        )
    }

    pub fn sin(&self) -> Self {
        let sin = self.scalar.sin();
        self.ad.create_unary_composite(
            self, sin, self.scalar.cos(),
        )
    }

    pub fn cos(&self) -> Self {
        let cos = self.scalar.cos();
        self.ad.create_unary_composite(
            self, cos, -self.scalar.sin(),
        )
    }

    pub fn tan(&self) -> Self {
        let tan = self.scalar.tan();
        self.ad.create_unary_composite(
            self, tan, 1.0 + tan * tan,
        )
    }

    pub fn asin(&self) -> Self {
        let asin = self.scalar.asin();
        self.ad.create_unary_composite(
            self, asin, 1.0 / (1.0 - self.scalar * self.scalar).sqrt(),
        )
    }

    pub fn acos(&self) -> Self {
        let acos = self.scalar.acos();
        self.ad.create_unary_composite(
            self, acos, -1.0 / (1.0 - self.scalar * self.scalar).sqrt(),
        )
    }

    pub fn atan(&self) -> Self {
        let atan = self.scalar.atan();
        self.ad.create_unary_composite(
            self, atan, 1.0 / (1.0 + self.scalar * self.scalar),
        )
    }

    pub fn sinh(&self) -> Self {
        let sinh = self.scalar.sinh();
        self.ad.create_unary_composite(
            self, sinh, self.scalar.cosh(),
        )
    }

    pub fn cosh(&self) -> Self {
        let cosh = self.scalar.cosh();
        self.ad.create_unary_composite(
            self, cosh, self.scalar.sinh(),
        )
    }

    pub fn tanh(&self) -> Self {
        let tanh = self.scalar.tanh();
        self.ad.create_unary_composite(
            self, tanh, 1.0 - tanh * tanh,
        )
    }

    pub fn asinh(&self) -> Self {
        let asinh = self.scalar.asinh();
        self.ad.create_unary_composite(
            self, asinh, 1.0 / (self.scalar + self.scalar * self.scalar).sqrt(),
        )
    }

    pub fn acosh(&self) -> Self {
        let acosh = self.scalar.acosh();
        self.ad.create_unary_composite(
            self, acosh, 1.0 / (self.scalar * self.scalar - 1.0).sqrt(),
        )
    }

    pub fn atanh(&self) -> Self {
        let atanh = self.scalar.atanh();
        self.ad.create_unary_composite(
            self, atanh, 1.0 / (1.0 - self.scalar * self.scalar),
        )
    }

    pub fn sqrt(&self) -> Self {
        let sqrt = self.scalar.sqrt();
        self.ad.create_unary_composite(
            self, sqrt, 0.5 / sqrt,
        )
    }

    pub fn cbrt(&self) -> Self {
        let cbrt = self.scalar.cbrt();
        self.ad.create_unary_composite(
            self, cbrt, 1.0 / 3.0 * (self.scalar / cbrt).powi(2),
        )
    }

    pub fn abs(&self) -> Self {
        let abs = self.scalar.abs();
        self.ad.create_unary_composite(
            self, abs, if self.scalar < 0.0 { -1.0 } else { 1.0 },
        )
    }

    pub fn relu(&self) -> Self {
        let relu = self.scalar.max(0.0);
        self.ad.create_unary_composite(
            self, relu, if self.scalar > 0.0 { 1.0 } else { 0.0 },
        )
    }

    pub fn leaky_relu(&self, alpha: f32) -> Self {
        let leaky_relu = self.scalar.max(0.0).max(self.scalar * alpha);
        self.ad.create_unary_composite(
            self, leaky_relu, if self.scalar > 0.0 { 1.0 } else { alpha },
        )
    }
}

impl <'a> NumberLike for ADNumber<'a> {
}

impl std::fmt::Debug for AD {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "AD {{ tapes: {:#?} }}", self.tapes.borrow())
    }
}

impl AD {
    pub fn new() -> AD {
        AD {
            tapes: RefCell::new(vec![(1, Tape::new())].into_iter().collect()),
            max_tape_id: RefCell::new(0),
            gradients: RefCell::new(HashMap::new()),
        }
    }

    pub fn reset(&self) {
        self.max_tape_id.replace(0);
        self.gradients.replace(HashMap::new());
        self.tapes.replace(HashMap::new());
    }

    fn next_tape_id(&self) -> usize {
        let mut id = self.max_tape_id.borrow_mut();
        *id += 1;
        *id
    }

    fn current_tape_id(&self) -> usize {
        if *self.max_tape_id.borrow() == 0 {
            self.next_tape_id()
        } else {
            *self.max_tape_id.borrow()
        }
    }

    fn get_tape_mut(&mut self) -> &mut Tape {
        let tape_id = *self.max_tape_id.borrow();
        let tape = self.tapes.get_mut().get_mut(&tape_id).unwrap();
        tape
    }

    pub fn create_constant(&self, scalar: f32) -> ADNumber {
        ADNumber {
            ad: self,
            id: 0,
            tape_id: 0,
            scalar,
        }
    }

    pub fn create_variable(&self, scalar: f32) -> ADNumber {
        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();

        let id = tape.records.len();
        tape.records.push(TapeRecord::new());

        let value = ADNumber {
            ad: self,
            id,
            tape_id,
            scalar,
        };

        value
    }

    pub fn create_binary_composite(
        &self,
        left: &ADNumber,
        right: &ADNumber,
        result: f32,
        diff_wrt_left: f32,
        diff_wrt_right: f32,
    ) -> ADNumber {
        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();

        if left.is_constant() && right.is_constant() {
            return self.create_constant(result);
        }

        let mut rec = TapeRecord::new();
        if left.is_variable() {
            rec.record(left, diff_wrt_left);
        }

        if right.is_variable() {
            rec.record(right, diff_wrt_right);
        }

        let id = tape.records.len();
        tape.records.push(rec);

        ADNumber {
            ad: self,
            id,
            tape_id,
            scalar: result,
        }
    }

    pub fn create_unary_composite(
        &self,
        operand: &ADNumber,
        result: f32,
        diff_wrt_operand: f32,
    ) -> ADNumber {
        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();
        let id = tape.records.len();

        if operand.is_constant() {
            return ADNumber {
                ad: self,
                id: 0,
                tape_id: 0,
                scalar: result,
            };
        }

        let mut rec = TapeRecord::new();
        rec.record(
            operand, diff_wrt_operand,
        );
        tape.records.push(rec);

        ADNumber {
            ad: self,
            id,
            tape_id,
            scalar: result,
        }
    }

    fn compute_gradient(&self, y: &ADNumber) -> Vec<f32> {
        let tape = &self.tapes.borrow()[&y.tape_id];
        let mut dy = vec![0.0; y.id+1];
        dy[y.id] = 1.0;

        // - go through each computation backwards
        // - increment gradient for each variable the computation depends on
        for (i, record) in tape.records.iter().enumerate().rev() {
            for partial in &record.partials {
                dy[partial.with_respect_to_id] += partial.differential * dy[i];
            }
        }

        dy
    }

    pub fn diff(&self, y: &ADNumber, wrt: &ADNumber) -> f32 {
        let mut gradients = self.gradients.borrow_mut();
        match gradients.get(&y.id) {
            Some(dy) => dy[wrt.id],
            None => {
                let dy = self.compute_gradient(y);
                let res = dy[wrt.id];
                gradients.insert(y.id, dy);
                res
            }
        }
    }
}

impl <'a> ops::Add<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn add(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.create_binary_composite(
            &self, &other,
            self.scalar + other.scalar,
            1.0, 1.0,
        )
    }
}

impl <'a> ops::Add<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn add(self, other: f32) -> ADNumber<'a> {
        self.ad.create_unary_composite(&self, self.scalar + other, 1.0)
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
        self.ad.create_binary_composite(
            &self, &other,
            self.scalar - other.scalar,
            1.0, -1.0,
        )
    }
}

impl <'a> ops::Sub<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn sub(self, other: f32) -> ADNumber<'a> {
        self.ad.create_unary_composite(&self, self.scalar - other, 1.0)
    }
}

impl <'a> ops::SubAssign<ADNumber<'a>> for ADNumber<'a> {
    fn sub_assign(&mut self, other: ADNumber<'a>) {
        *self = *self + other;
    }
}

impl <'a> ops::Mul<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn mul(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.create_binary_composite(
            &self, &other,
            self.scalar * other.scalar,
            other.scalar, self.scalar,
        )
    }
}

impl <'a> ops::Mul<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn mul(self, other: f32) -> ADNumber<'a> {
        self.ad.create_unary_composite(&self, self.scalar * other, other)
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
        self.ad.create_binary_composite(
            &self, &other,
            self.scalar / other.scalar,
            1.0 / other.scalar, -self.scalar / other.scalar.powi(2),
        )
    }
}

impl <'a> ops::Div<f32> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn div(self, other: f32) -> ADNumber<'a> {
        self.ad.create_unary_composite(&self, self.scalar / other, 1.0 / other)
    }
}

impl <'a> ops::DivAssign<ADNumber<'a>> for ADNumber<'a> {
    fn div_assign(&mut self, other: ADNumber<'a>) {
        *self = *self / other;
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x + x;
        println!("{:#?}", ad);
        assert_eq!(y.diff(&x), 2.0);
        assert_eq!((y + ad.create_variable(7.0)).diff(&x), 2.0);
        assert_eq!((y + ad.create_constant(7.0)).diff(&x), 2.0);
        assert_eq!((y + 7.0).diff(&x), 2.0);
        assert_eq!((y + 7.0).scalar, 11.0);
    }

    #[test]
    fn test_mul_sub() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x - x;
        let z = ad.create_variable(3.0);
        let t = x * z - y;
        println!("{:#?}", ad);
        assert_eq!(t.diff(&x), 3.0);
        assert_eq!(t.diff(&z), 2.0);
        assert_eq!(t.diff(&y), -1.0);
    }
}
