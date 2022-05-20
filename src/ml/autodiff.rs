mod computation;

use std::collections::hash_map::{HashMap};

use crate::ml::{
    NumberFactory,
    DifferentiableNumberFactory,
    NumberLike,
    NeuronActivation,
};

pub use computation::{
    ComputationEnvironment,
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
    pushed: usize,
    record: Record,
}

impl DiffDefinerHelper {
    fn new() -> Self {
        DiffDefinerHelper {
            pushed: 0,
            record: Default::default(),
        }
    }

    fn diff(&mut self, variable: &ADNumber, diff: f32) -> &mut Self {
        if let Some(id) = variable.id {
            self.pushed += 1;
            self.record.partials.push(PartialDiff {
                with_respect_to_id: id,
                diff,
            });
        }
        self
    }
}

pub struct TapeRecordResult {
    next_number_id: Option<usize>,
}

impl Default for TapeRecordResult {
    fn default() -> Self {
        Self { next_number_id: None }
    }
}

impl TapeRecordResult {
    pub fn new() { Default::default() }

    pub fn result(&self, scalar: f32) -> ADNumber {
        ADNumber::new(self.next_number_id, scalar)
    }
}

#[derive(Default)]
struct Tape {
    records: Vec<Record>
}

impl Tape {
    fn record<D: FnOnce(&mut DiffDefinerHelper)>(&mut self, definer: D) -> TapeRecordResult {
        let mut log = DiffDefinerHelper::new();
        definer(&mut log);
        let next_number_id = if log.pushed > 0 {
            self.records.push(log.record);
            Some(self.records.len() -1)
        } else {
            None
        };
        TapeRecordResult { next_number_id }
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn push_empty_record(&mut self) -> &mut Self {
        self.records.push(Default::default());
        self
    }

    fn compute_gradient(&self, y: &ADNumber) -> Vec<f32> {
        if y.id.is_none() {
            panic!("cannot take the gradient of a constant");
        }

        let y_id = y.id.expect("y should be a variable");

        let mut gradient = vec![0.0; y_id + 1];
        gradient[y_id] = 1.0;

        for i in (0..y_id+1).rev() {
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

    fn constant(&mut self, scalar: f32) -> ADNumber {
        ADNumber::new(None, scalar)
    }
}

impl DifferentiableNumberFactory<ADNumber> for AutoDiff {
    fn diff(&mut self, y: &ADNumber, x: &ADNumber) -> f32 {
        if x.id.is_none() {
            // The diff wrt a constant is always zero.
            return 0.0;
        }

        let x_id = x.id.expect("x should be a variable");

        if let Some(y_id) = y.id {
            match self.gradients.get(&y_id) {
                Some(gradient) => gradient[x_id],
                None => {
                    let gradient = self.tape.compute_gradient(&y);
                    let diff = gradient[x_id];
                    self.gradients.insert(y_id, gradient);
                    diff
                }
            }
        } else {
            // The diff of a constant is always zero.
            return 0.0
        }
    }

    fn compose(&mut self, result: f32, partials: Vec<(&ADNumber, f32)>) -> ADNumber {
        self.tape.record(|log| {
            for (n, d) in partials {
                log.diff(n, d);
            }
        }).result(result)
    }

    fn variable(&mut self, scalar: f32) -> ADNumber {
        let id = Some(self.tape.len());
        self.tape.push_empty_record();
        ADNumber::new(id, scalar)
    }
}

#[derive(Copy, Clone)]
pub struct ADNumber {
    id: Option<usize>,
    scalar: f32,
}

impl ADNumber {
    pub fn new(id: Option<usize>, scalar: f32) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_simple() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(1.0);
        let y = ad.add(&x, &x);
        let dy_dx = ad.diff(&y, &x);
        assert_eq!(y.scalar(), 2.0);
        assert_eq!(dy_dx, 2.0);
    }

    #[test]
    fn test_dx2_dx() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(2.0);
        let y = ad.mul(&x, &x);
        let dy_dx = ad.diff(&y, &x);
        assert_eq!(dy_dx, 4.0);
        assert_eq!(y.scalar(), 4.0);
    }

    #[test]
    fn test_dx2y_dx_dx2y_dy() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(2.0);
        let y = ad.variable(3.0);
        let pz = ad.mul(&x, &x);
        let z = ad.mul(&y, &pz);

        assert_eq!(ad.diff(&z, &x), 12.0);
        assert_eq!(ad.diff(&z, &y), 4.0);
    }

    #[test]
    fn test_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(1.0);
        let y = ad.variable(2.0);
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
        let x = ad.variable(3.0);
        let y = ad.variable(2.0);
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
        let x = ad.variable(1.0);
        let y = ad.variable(2.0);
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
        let x = ad.variable(1.0);
        let y = ad.exp(&x);
        let dy_dx = ad.diff(&y, &x);

        assert_eq!(dy_dx, 1.0f32.exp());
    }

    #[test]
    fn test_much_more_complex_diff() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(3.0);
        let y = ad.variable(4.0);
        let exp_x = ad.exp(&x);
        let exp_x_minus_y = ad.sub(&exp_x, &y);
        let o = ad.div(&y, &exp_x_minus_y);

        assert_eq!(ad.diff(&o, &x), -0.310507656);
        assert_eq!(ad.diff(&o, &y), 0.077626914);
    }

    #[test]
    fn test_relu() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(4.0);
        let y = ad.activate_neuron(&x, &NeuronActivation::ReLu);
        assert_eq!(ad.diff(&y, &x), 1.0);
    }

    #[test]
    fn test_leaky_relu() {
        let mut ad = AutoDiff::new();
        let x = ad.variable(-4.0);
        let y = ad.activate_neuron(&x, &NeuronActivation::LeakyRelu(0.1));
        assert_eq!(ad.diff(&y, &x), 0.1);
    }
}
