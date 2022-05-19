use rand::prelude::*;
use std::collections::HashMap;

use crate::ml::{
    NumberFactory,
    DifferentiableNumberFactory,
    ADNumber,
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

#[derive(Debug)]
struct PartialDerivative {
    with_respect_to_id: usize,
    diff: f32,
}

#[derive(Debug)]
struct Record {
    partials: Vec<PartialDerivative>,
}

impl Record {
    fn new() -> Self {
        Self {
            partials: Vec::new(),
        }
    }
}
#[derive(Debug)]
struct Tape {
    records: Vec<Record>,
}

impl Tape {
    fn new() -> Self {
        Self {
            records: Vec::new()
        }
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn push_empty_record(&mut self) -> &mut Self {
        self.records.push(Record::new());
        self
    }
}

#[derive(Debug)]
pub struct ADFactory {
    rng: ThreadRng,
    tape: Tape,
    gradients: HashMap<usize, Vec<f32>>,
}

impl ADFactory {
    pub fn new() -> Self {
        Self {
            rng: thread_rng(),
            tape: Tape::new(),
            gradients: HashMap::new(),
        }
    }

    fn binary_operation(
        &mut self,
        left: ADNumber, right: ADNumber,
        result: f32,
        diff_left: f32, diff_right: f32,
    ) -> ADNumber {
        let id = self.tape.len();

        let partials = vec![
            PartialDerivative {
                with_respect_to_id: left.id(),
                diff: diff_left,
            },
            PartialDerivative {
                with_respect_to_id: right.id(),
                diff: diff_right,
            },
        ];

        let record = Record {
            partials,
        };

        self.tape.records.push(record);

        ADNumber::new(id, result)
    }

    fn unary_operation(
        &mut self,
        operand: ADNumber,
        result: f32,
        diff: f32,
    ) -> ADNumber {
        let id = self.tape.len();

        let partials = vec![
            PartialDerivative {
                with_respect_to_id: operand.id(),
                diff,
            },
        ];

        let record = Record {
            partials,
        };

        self.tape.records.push(record);

        ADNumber::new(id, result)
    }

    fn compute_gradient(&self, y: ADNumber) -> Vec<f32> {
        let mut gradient = vec![0.0; y.id() + 1];
        gradient[y.id()] = 1.0;

        for i in (0..y.id()+1).rev() {
            let record = &self.tape.records[i];
            for partial in &record.partials {
                gradient[partial.with_respect_to_id] += gradient[i] * partial.diff;
            }
        }

        gradient
    }

    fn diff(&mut self, y: ADNumber, x: ADNumber) -> f32 {
        match self.gradients.get(&y.id()) {
            Some(gradient) => gradient[x.id()],
            None => {
                let gradient = self.compute_gradient(y);
                let diff = gradient[x.id()];
                self.gradients.insert(y.id(), gradient);
                diff
            }
        }
    }
}

impl NumberFactory<ADNumber> for ADFactory {
    fn create_variable(&mut self, scalar: f32) -> ADNumber {
        let id = self.tape.len();
        self.tape.push_empty_record();
        ADNumber::new(id, scalar)
    }

    fn create_random_variable(&mut self) -> ADNumber {
        let scalar = self.rng.gen();
        self.create_variable(scalar)
    }

    fn multiply(&mut self, left: ADNumber, right: ADNumber) -> ADNumber {
        self.binary_operation(
            left, right,
            left.scalar() * right.scalar(),
            right.scalar(), left.scalar()
        )
    }

    fn divide(&mut self, left: ADNumber, right: ADNumber) -> ADNumber {
        self.binary_operation(
            left, right,
            left.scalar() / right.scalar(),
            1.0 / right.scalar(), -left.scalar() / right.scalar().powi(2)
        )
    }

    fn addition(&mut self, left: ADNumber, right: ADNumber) -> ADNumber {
        self.binary_operation(
            left, right,
            left.scalar() + right.scalar(),
            1.0, 1.0,
        )
    }

    fn subtract(&mut self, left: ADNumber, right: ADNumber) -> ADNumber {
        self.binary_operation(
            left, right,
            left.scalar() - right.scalar(),
            1.0, -1.0,
        )
    }

    fn exp(&mut self, operand: ADNumber) -> ADNumber {
        self.unary_operation(
            operand,
            operand.scalar().exp(),
            operand.scalar().exp(),
        )
    }

    fn binary_operation(
        &mut self,
        left: ADNumber, right: ADNumber,
        result: f32,
        diff_left: f32, diff_right: f32,
    ) -> ADNumber {
        self.binary_operation(left, right, result, diff_left, diff_right)
    }

    fn unary_operation(
        &mut self,
        operand: ADNumber,
        result: f32,
        diff: f32,
    ) -> ADNumber {
        self.unary_operation(operand, result, diff)
    }

    fn to_scalar(&self, operand: ADNumber) -> f32 {
        operand.scalar()
    }
}

impl DifferentiableNumberFactory<ADNumber> for ADFactory {
    fn diff(&mut self, y: ADNumber, x: ADNumber) -> f32 {
        self.diff(y, x)
    }
}


#[test]
fn test_add_simple() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(1.0);
    let y = ad.addition(x, x);
    let dy_dx = ad.diff(y, x);
    assert_eq!(y.scalar(), 2.0);
    assert_eq!(dy_dx, 2.0);
}

#[test]
fn test_dx2_dx() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(2.0);
    let y = ad.multiply(x, x);
    let dy_dx = ad.diff(y, x);
    assert_eq!(dy_dx, 4.0);
    assert_eq!(y.scalar(), 4.0);
}

#[test]
fn test_dx2y_dx_dx2y_dy() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(2.0);
    let y = ad.create_variable(3.0);
    let pz = ad.multiply(x, x);
    let z = ad.multiply(y, pz);

    assert_eq!(ad.diff(z, x), 12.0);
    assert_eq!(ad.diff(z, y), 4.0);
}

#[test]
fn test_sub() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(1.0);
    let y = ad.create_variable(2.0);
    let z = ad.subtract(x, y);
    let dz_dx = ad.diff(z, x);
    let dz_dy = ad.diff(z, y);

    assert_eq!(dz_dx, 1.0);
    assert_eq!(dz_dy, -1.0);
    assert_eq!(z.scalar(), -1.0);
}

#[test]
fn test_mul() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(3.0);
    let y = ad.create_variable(2.0);
    let z = ad.multiply(x, y);
    let dz_dx = ad.diff(z, x);
    let dz_dy = ad.diff(z, y);

    assert_eq!(dz_dx, 2.0);
    assert_eq!(dz_dy, 3.0);
    assert_eq!(z.scalar(), 6.0);
}

#[test]
fn test_div() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(1.0);
    let y = ad.create_variable(2.0);
    let z = ad.divide(x, y);
    let dz_dx = ad.diff(z, x);
    let dz_dy = ad.diff(z, y);

    assert_eq!(dz_dx, 0.5);
    assert_eq!(dz_dy, -0.25);
    assert_eq!(z.scalar(), 0.5);
}

#[test]
fn test_exp() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(1.0);
    let y = ad.exp(x);
    let dy_dx = ad.diff(y, x);

    assert_eq!(dy_dx, 1.0f32.exp());
}

#[test]
fn test_much_more_complex_diff() {
    let mut ad = ADFactory::new();
    let x = ad.create_variable(3.0);
    let y = ad.create_variable(4.0);
    let exp_x = ad.exp(x);
    let exp_x_minus_y = ad.subtract(exp_x, y);
    let o = ad.divide(y, exp_x_minus_y);

    assert_eq!(ad.diff(o, x), -0.310507656);
    assert_eq!(ad.diff(o, y), 0.077626914);
}
