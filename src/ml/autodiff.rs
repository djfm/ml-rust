use std::collections::hash_map::{HashMap};

use crate::ml::{
    NumberFactory,
    NumberLike,
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

struct PartialDiff {
    with_respect_to_id: usize,
    diff: f32,
}

#[derive(Default)]
struct Record {
    partials: Vec<PartialDiff>
}

struct DefinerHelper {
    record: Record,
}

impl DefinerHelper {
    fn new() -> Self {
        DefinerHelper {
            record: Default::default(),
        }
    }

    fn diff(&mut self, variable: &ADNumber, diff: f32) -> &mut Self {
        self.record.partials.push(PartialDiff {
            with_respect_to_id: variable.id,
            diff,
        });
        self
    }
}

#[derive(Default)]
struct Tape {
    records: Vec<Record>
}

impl Tape {
    fn new() -> Self {
        Default::default()
    }

    fn record<D: FnOnce(&mut DefinerHelper)>(&mut self, definer: D) -> &mut Self {
        let mut log = DefinerHelper::new();
        definer(&mut log);
        self.records.push(log.record);
        self
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn push_empty_record(&mut self) -> &mut Self {
        self.records.push(Default::default());
        self
    }

    fn result(&self, scalar: f32) -> ADNumber {
        ADNumber::new(self.len() - 1, scalar)
    }

    fn compute_gradient(&self, y: &ADNumber) -> Vec<f32> {
        let mut gradient = vec![0.0; y.id + 1];
        gradient[y.id] = 1.0;

        for i in (0..y.id+1).rev() {
            let record = &self.records[i];
            for partial in &record.partials {
                gradient[partial.with_respect_to_id] += partial.diff * gradient[i];
            }
        }

        gradient
    }
}

#[derive(Default)]
pub struct Autodiff {
    tape: Tape,
    gradients: HashMap<usize, Vec<f32>>,
}

impl NumberFactory<ADNumber> for Autodiff {
    fn new() -> Self { Default::default() }

    fn diff(&mut self, y: &ADNumber, x: &ADNumber) -> f32 {
        match self.gradients.get(&y.id) {
            Some(gradient) => gradient[x.id],
            None => {
                let gradient = self.tape.compute_gradient(&y);
                let diff = gradient[x.id];
                self.gradients.insert(y.id, gradient);
                diff
            }
        }
    }

    fn from_scalar(&mut self, scalar: f32) -> ADNumber {
        self.tape.push_empty_record();
        ADNumber::new(self.tape.len() - 1, scalar)
    }

    fn add(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, 1.0).diff(b, 1.0);
        }).result(a.scalar + b.scalar)
    }

    fn sub(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, 1.0).diff(b, -1.0);
        }).result(a.scalar - b.scalar)
    }

    fn mul(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, b.scalar).diff(b, a.scalar);
        }).result(a.scalar * b.scalar)
    }

    fn div(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, 1.0 / b.scalar).diff(b, -a.scalar / b.scalar.powi(2));
        }).result(a.scalar / b.scalar)
    }

    fn exp(&mut self, a: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, a.scalar.exp());
        }).result(a.scalar.exp())
    }

    fn log(&mut self, a: &ADNumber) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, 1.0 / a.scalar);
        }).result(a.scalar.ln())
    }

    fn powi(&mut self, a: &ADNumber, n: i32) -> ADNumber {
        self.tape.record(|mut log| {
            log.diff(a, n as f32 * a.scalar.powi(n - 1));
        }).result(a.scalar.powi(n))
    }

    fn pow(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        // a^b = e^(b * ln(a))
        self.tape.record(|mut log| {
            log
                .diff(a, a.scalar.powf(b.scalar - 1.0))
                .diff(b, a.scalar.ln() * a.scalar.powf(b.scalar));
        }).result(a.scalar.powf(b.scalar))
    }

    fn activate_neuron(&mut self, a: &ADNumber, activation: &NeuronActivation) -> ADNumber {
        match activation {
            NeuronActivation::None => *a,
            NeuronActivation::ReLu => {
                if a.scalar() > 0.0 {
                    self.tape.record(|mut log| {
                        log.diff(a, 1.0);
                    }).result(a.scalar())
                } else {
                    self.tape.record(|mut log| {
                        log.diff(a, 0.0);
                    }).result(0.0)
                }
            }
            NeuronActivation::LeakyRelu(alpha) => {
                if a.scalar() > 0.0 {
                    self.tape.record(|mut log| {
                        log.diff(a, 1.0);
                    }).result(a.scalar())
                } else {
                    self.tape.record(|mut log| {
                        log.diff(a, *alpha);
                    }).result(0.0)
                }
            }
        }
    }

    fn activate_layer(&mut self, a: &[ADNumber], activation: &LayerActivation) -> Vec<ADNumber> {
        match activation {
            LayerActivation::None => a.to_vec(),
            LayerActivation::SoftMax => {
                let mut sum = self.from_scalar(0.0);
                let mut res = Vec::with_capacity(a.len());

                for (i, v) in a.iter().enumerate() {
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

    fn compute_error(&mut self, expected: &[ADNumber], actual: &[ADNumber], error_function: &ErrorFunction) -> ADNumber {
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
                    let log = self.log(&a);
                    let mul = self.mul(&log, &e);
                    sum = self.sub(&sum, &mul);
                }
                sum
            },
        }
    }
}

#[derive(Copy, Clone)]
pub struct ADNumber {
    id: usize,
    scalar: f32,
}

impl ADNumber {
    fn new(id: usize, scalar: f32) -> Self {
        ADNumber {
            id,
            scalar,
        }
    }
}

impl NumberLike for ADNumber {
    fn scalar(&self) -> f32 {
        self.scalar
    }
}

#[test]
fn test_add_simple() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(1.0);
    let y = ad.add(&x, &x);
    let dy_dx = ad.diff(&y, &x);
    assert_eq!(y.scalar(), 2.0);
    assert_eq!(dy_dx, 2.0);
}

#[test]
fn test_dx2_dx() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(2.0);
    let y = ad.mul(&x, &x);
    let dy_dx = ad.diff(&y, &x);
    assert_eq!(dy_dx, 4.0);
    assert_eq!(y.scalar(), 4.0);
}

#[test]
fn test_dx2y_dx_dx2y_dy() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(2.0);
    let y = ad.from_scalar(3.0);
    let pz = ad.mul(&x, &x);
    let z = ad.mul(&y, &pz);

    assert_eq!(ad.diff(&z, &x), 12.0);
    assert_eq!(ad.diff(&z, &y), 4.0);
}

#[test]
fn test_sub() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(1.0);
    let y = ad.from_scalar(2.0);
    let z = ad.sub(&x, &y);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);

    assert_eq!(dz_dx, 1.0);
    assert_eq!(dz_dy, -1.0);
    assert_eq!(z.scalar(), -1.0);
}

#[test]
fn test_mul() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(3.0);
    let y = ad.from_scalar(2.0);
    let z = ad.mul(&x, &y);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);

    assert_eq!(dz_dx, 2.0);
    assert_eq!(dz_dy, 3.0);
    assert_eq!(z.scalar(), 6.0);
}

#[test]
fn test_div() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(1.0);
    let y = ad.from_scalar(2.0);
    let z = ad.div(&x, &y);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);

    assert_eq!(dz_dx, 0.5);
    assert_eq!(dz_dy, -0.25);
    assert_eq!(z.scalar(), 0.5);
}

#[test]
fn test_exp() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(1.0);
    let y = ad.exp(&x);
    let dy_dx = ad.diff(&y, &x);

    assert_eq!(dy_dx, 1.0f32.exp());
}

#[test]
fn test_much_more_complex_diff() {
    let mut ad = Autodiff::new();
    let x = ad.from_scalar(3.0);
    let y = ad.from_scalar(4.0);
    let exp_x = ad.exp(&x);
    let exp_x_minus_y = ad.sub(&exp_x, &y);
    let o = ad.div(&y, &exp_x_minus_y);

    assert_eq!(ad.diff(&o, &x), -0.310507656);
    assert_eq!(ad.diff(&o, &y), 0.077626914);
}
