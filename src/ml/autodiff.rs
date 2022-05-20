use std::collections::hash_map::{HashMap};

use crate::ml::{
    NumberFactory,
    DifferentiableNumberFactory,
    NumberLike,
    NeuronActivation,
};

struct PartialDiff {
    with_respect_to_id: usize,
    diff: f32,
}

#[derive(Default)]
struct Record {
    partials: Vec<PartialDiff>
}

pub struct DiffDefinerHelper {
    record: Record,
}

impl DiffDefinerHelper {
    fn new() -> Self {
        DiffDefinerHelper {
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
    fn record<D: FnOnce(&mut DiffDefinerHelper)>(&mut self, definer: D) -> &mut Self {
        let mut log = DiffDefinerHelper::new();
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
pub struct AutoDiff {
    tape: Tape,
    gradients: HashMap<usize, Vec<f32>>,
}

impl AutoDiff {
    pub fn new() -> Self {
        Default::default()
    }
}

impl NumberFactory<ADNumber> for AutoDiff {
    fn get_as_differentiable(&mut self) -> Option<&mut (dyn DifferentiableNumberFactory<ADNumber>)> {
        Some(self)
    }

    fn from_scalar(&mut self, scalar: f32) -> ADNumber {
        self.tape.push_empty_record();
        ADNumber::new(self.tape.len() - 1, scalar)
    }

    fn add(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, 1.0).diff(b, 1.0);
        }).result(a.scalar + b.scalar)
    }

    fn sub(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, 1.0).diff(b, -1.0);
        }).result(a.scalar - b.scalar)
    }

    fn mul(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, b.scalar).diff(b, a.scalar);
        }).result(a.scalar * b.scalar)
    }

    fn div(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, 1.0 / b.scalar).diff(b, -a.scalar / b.scalar.powi(2));
        }).result(a.scalar / b.scalar)
    }

    fn exp(&mut self, a: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, a.scalar.exp());
        }).result(a.scalar.exp())
    }

    fn ln(&mut self, a: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, 1.0 / a.scalar);
        }).result(a.scalar.ln())
    }

    fn powi(&mut self, a: &ADNumber, n: i32) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, n as f32 * a.scalar.powi(n - 1));
        }).result(a.scalar.powi(n))
    }

    fn pow(&mut self, a: &ADNumber, b: &ADNumber) -> ADNumber {
        // a^b = e^(b * ln(a))
        self.tape.record(|log| {
            log
                .diff(a, a.scalar.powf(b.scalar - 1.0))
                .diff(b, a.scalar.ln() * a.scalar.powf(b.scalar));
        }).result(a.scalar.powf(b.scalar))
    }

    fn neg(&mut self, a: &ADNumber) -> ADNumber {
        self.tape.record(|log| {
            log.diff(a, -1.0);
        }).result(-a.scalar)
    }
}

impl DifferentiableNumberFactory<ADNumber> for AutoDiff {
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

    fn compose(&mut self, result: f32, partials: Vec<(&ADNumber, f32)>) -> ADNumber {
        let id = self.tape.len();

        self.tape.record(|log| {
            for (n, d) in partials {
                log.diff(n, d);
            }
        });

        ADNumber::new(id, result)
    }
}

#[derive(Copy, Clone)]
pub struct ADNumber {
    id: usize,
    scalar: f32,
}

impl ADNumber {
    pub fn new(id: usize, scalar: f32) -> Self {
        ADNumber {
            id,
            scalar,
        }
    }

    pub fn scalar(&self) -> f32 {
        self.scalar
    }
}

impl NumberLike for ADNumber {
    fn scalar(&self) -> f32 {
        self.scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_simple() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(1.0);
        let y = ad.add(&x, &x);
        let dy_dx = ad.diff(&y, &x);
        assert_eq!(y.scalar(), 2.0);
        assert_eq!(dy_dx, 2.0);
    }

    #[test]
    fn test_dx2_dx() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(2.0);
        let y = ad.mul(&x, &x);
        let dy_dx = ad.diff(&y, &x);
        assert_eq!(dy_dx, 4.0);
        assert_eq!(y.scalar(), 4.0);
    }

    #[test]
    fn test_dx2y_dx_dx2y_dy() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(2.0);
        let y = ad.from_scalar(3.0);
        let pz = ad.mul(&x, &x);
        let z = ad.mul(&y, &pz);

        assert_eq!(ad.diff(&z, &x), 12.0);
        assert_eq!(ad.diff(&z, &y), 4.0);
    }

    #[test]
    fn test_sub() {
        let mut ad = AutoDiff::new();
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
        let mut ad = AutoDiff::new();
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
        let mut ad = AutoDiff::new();
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
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(1.0);
        let y = ad.exp(&x);
        let dy_dx = ad.diff(&y, &x);

        assert_eq!(dy_dx, 1.0f32.exp());
    }

    #[test]
    fn test_much_more_complex_diff() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(3.0);
        let y = ad.from_scalar(4.0);
        let exp_x = ad.exp(&x);
        let exp_x_minus_y = ad.sub(&exp_x, &y);
        let o = ad.div(&y, &exp_x_minus_y);

        assert_eq!(ad.diff(&o, &x), -0.310507656);
        assert_eq!(ad.diff(&o, &y), 0.077626914);
    }

    #[test]
    fn test_relu() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(4.0);
        let y = ad.activate_neuron(&x, &NeuronActivation::ReLu);
        assert_eq!(ad.diff(&y, &x), 1.0);
    }

    #[test]
    fn test_leaky_relu() {
        let mut ad = AutoDiff::new();
        let x = ad.from_scalar(-4.0);
        let y = ad.activate_neuron(&x, &NeuronActivation::LeakyRelu(0.1));
        assert_eq!(ad.diff(&y, &x), 0.1);
    }
}
