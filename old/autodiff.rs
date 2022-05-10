use std::collections::HashMap;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

#[derive(Copy, Clone, Debug)]
pub struct ADValue {
    id: Option<usize>,
    pub value: f32,
}

impl ADValue {
    pub fn is_constant(&self) -> bool {
        self.id.is_none()
    }

    pub fn is_variable(&self) -> bool {
        self.id.is_some()
    }
}

impl std::cmp::PartialEq for ADValue {
    fn eq(&self, other: &ADValue) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for ADValue {
    fn partial_cmp(&self, other: &ADValue) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

#[derive(Debug, Copy, Clone)]
struct PartialDiff {
    with_respect_to_id: usize,
    value: f32,
}

#[derive(Debug, Clone)]
struct TapeRecord {
    partials: Vec<PartialDiff>,
}

#[derive(Debug, Clone)]
struct Tape {
    records: Vec<TapeRecord>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape {
            records: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}

#[derive(Debug, Clone)]
pub struct AutoDiff {
    tape: Tape,
    gradients: HashMap<usize, Vec<f32>>,
}

impl AutoDiff {
    pub fn new() -> AutoDiff {
        AutoDiff {
            tape: Tape::new(),
            gradients: HashMap::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.tape.len()
    }

    pub fn reset(&mut self) {
        self.tape = Tape::new();
        self.gradients.clear();
    }

    pub fn create_variable(&mut self, value: f32) -> ADValue {
        let id = Some(self.tape.len());
        self.tape.records.push(TapeRecord {
            partials: Vec::new(),
        });
        ADValue {
            id,
            value,
        }
    }

    pub fn create_constant(&self, value: f32) -> ADValue {
        ADValue {
            id: None,
            value,
        }
    }

    fn compute_gradient(&mut self, y: ADValue) {
        if y.is_constant() {
            panic!("cannot compute gradient of a constant value");
        }

        let y_id = y.id.unwrap();
        let mut dy = vec![0.0; y_id + 1];

        dy[y_id] = 1.0;
        for i in (0..y_id+1).rev() {
            match self.tape.records.get(i) {
                Some(record) => {
                    for partial in &record.partials {
                        dy[partial.with_respect_to_id] += partial.value * dy[i];
                    }
                },
                None => {
                    panic!("partial derivative of expression not found on tape");
                },
            }
        }

        self.gradients.insert(y_id, dy);
    }

    pub fn diff(&mut self, y: ADValue, wrt: ADValue) -> f32 {
        if y.is_constant() {
            panic!("cannot compute gradient of a constant value");
        }

        if wrt.is_constant() {
            panic!("cannot compute gradient with respect to a constant value");
        }

        let y_id = y.id.unwrap();
        let wrt_id = wrt.id.unwrap();

        match self.gradients.get(&y_id) {
            Some(dy) => dy[wrt_id],
            None => {
                self.compute_gradient(y);
                self.gradients[&y_id][wrt_id]
            }
        }
    }

    pub fn add(&mut self, left: ADValue, right: ADValue) -> ADValue {
        if left.is_constant() && right.is_constant() {
            return self.create_constant(left.value + right.value);
        }

        let id = Some(self.tape.len());

        let mut partials = Vec::new();

        if left.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: left.id.unwrap(),
                value: 1.0,
            });
        }

        if right.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: right.id.unwrap(),
                value: 1.0,
            });
        }

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: left.value + right.value,
        }
    }

    pub fn sub(&mut self, left: ADValue, right: ADValue) -> ADValue {
        if left.is_constant() && right.is_constant() {
            return self.create_constant(left.value - right.value);
        }

        let id = Some(self.tape.len());

        let mut partials = Vec::new();

        if left.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: left.id.unwrap(),
                value: 1.0,
            });
        }

        if right.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: right.id.unwrap(),
                value: -1.0,
            });
        }

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: left.value - right.value,
        }
    }

    pub fn mul(&mut self, left: ADValue, right: ADValue) -> ADValue {
        if left.is_constant() && right.is_constant() {
            return self.create_constant(left.value * right.value);
        }

        let id = Some(self.tape.len());

        let mut partials = Vec::new();

        if left.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: left.id.unwrap(),
                value: right.value,
            });
        }

        if right.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: right.id.unwrap(),
                value: left.value,
            });
        }

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: left.value * right.value,
        }
    }

    pub fn div(&mut self, left: ADValue, right: ADValue) -> ADValue {
        if left.is_constant() && right.is_constant() {
            return self.create_constant(left.value / right.value);
        }

        let id = Some(self.tape.len());

        let mut partials = Vec::new();

        if left.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: left.id.unwrap(),
                value: 1.0 / right.value,
            });
        }

        if right.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: right.id.unwrap(),
                value: -left.value / right.value.powi(2),
            });
        }

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: left.value / right.value,
        }
    }

    pub fn exp(&mut self, v: ADValue) -> ADValue {
        let id = if v.is_variable() { Some(self.tape.len()) } else { None };
        let exp = v.value.exp();

        let mut partials = Vec::new();

        if v.is_variable() {
            partials.push(PartialDiff {
                with_respect_to_id: v.id.unwrap(),
                value: exp,
            });

            self.tape.records.push(TapeRecord {
                partials,
            });
        }

        ADValue {
            id,
            value: exp,
        }
    }

    pub fn apply_neuron_activation(&mut self, v: ADValue, activation: &NeuronActivation) -> ADValue {
        match activation {
            NeuronActivation::LeakyReLU(alpha) => {
                let id = Some(self.tape.len());

                let partials = vec![
                    PartialDiff {
                        with_respect_to_id: v.id.unwrap(),
                        value: if v.value > 0.0 {
                            1.0
                        } else {
                            *alpha
                        },
                    }
                ];

                self.tape.records.push(TapeRecord {
                    partials,
                });

                ADValue {
                    id,
                    value: if v.value > 0.0 {
                        v.value
                    } else {
                        v.value * *alpha
                    },
                }
            },
        }
    }

    pub fn apply_layer_activation(&mut self, values: &Vec<ADValue>, activation: &LayerActivation) -> Vec<ADValue> {
        match activation {
            LayerActivation::SoftMax => {
                let mut res = Vec::new();
                let mut sum = self.create_variable(0.0);

                for v in values.iter() {
                    let exp = self.exp(*v);
                    sum = self.add(sum, exp);
                    res.push(exp);
                }

                for r in res.iter_mut() {
                    *r = self.div(*r, sum);
                }

                res
            }
        }
    }

    pub fn euclidean_distance_squared(&mut self, left: &Vec<ADValue>, right: &Vec<ADValue>) -> ADValue {
        if left.len() != right.len() {
            panic!("cannot compute distance between vectors of different lengths");
        }

        let mut distance = self.create_variable(0.0);
        left.iter().zip(right.iter()).for_each(|(l, r)| {
            let diff = self.sub(*l, *r);
            let square = self.mul(diff, diff);
            distance = self.add(distance, square);
        });

        distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_simple() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(1.0);
        let y = ad.add(x, x);
        let dy_dx = ad.diff(y, x);
        assert_eq!(y.value, 2.0);
        assert_eq!(dy_dx, 2.0);
    }

    #[test]
    fn test_dx2_dx() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(2.0);
        let y = ad.mul(x, x);
        let dy_dx = ad.diff(y, x);
        assert_eq!(dy_dx, 4.0);
        assert_eq!(y.value, 4.0);
    }

    #[test]
    fn test_dx2y_dx_dx2y_dy() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(2.0);
        let y = ad.create_variable(3.0);
        let pz = ad.mul(x, x);
        let z = ad.mul(y, pz);

        assert_eq!(ad.diff(z, x), 12.0);
        assert_eq!(ad.diff(z, y), 4.0);
    }

    #[test]
    fn test_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(1.0);
        let y = ad.create_variable(2.0);
        let z = ad.sub(x, y);
        let dz_dx = ad.diff(z, x);
        let dz_dy = ad.diff(z, y);

        assert_eq!(dz_dx, 1.0);
        assert_eq!(dz_dy, -1.0);
        assert_eq!(z.value, -1.0);
    }

    #[test]
    fn test_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(3.0);
        let y = ad.create_variable(2.0);
        let z = ad.mul(x, y);
        let dz_dx = ad.diff(z, x);
        let dz_dy = ad.diff(z, y);

        assert_eq!(dz_dx, 2.0);
        assert_eq!(dz_dy, 3.0);
        assert_eq!(z.value, 6.0);
    }

    #[test]
    fn test_div() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(1.0);
        let y = ad.create_variable(2.0);
        let z = ad.div(x, y);
        let dz_dx = ad.diff(z, x);
        let dz_dy = ad.diff(z, y);

        assert_eq!(dz_dx, 0.5);
        assert_eq!(dz_dy, -0.25);
        assert_eq!(z.value, 0.5);
    }

    #[test]
    fn test_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(1.0);
        let y = ad.exp(x);
        let dy_dx = ad.diff(y, x);

        assert_eq!(dy_dx, 1.0f32.exp());
    }

    #[test]
    fn test_much_more_complex_diff() {
        let mut ad = AutoDiff::new();
        let x = ad.create_variable(3.0);
        let y = ad.create_variable(4.0);
        let exp_x = ad.exp(x);
        let exp_x_minus_y = ad.sub(exp_x, y);
        let o = ad.div(y, exp_x_minus_y);

        assert_eq!(ad.diff(o, x), -0.310507656);
        assert_eq!(ad.diff(o, y), 0.077626914);
    }
}
