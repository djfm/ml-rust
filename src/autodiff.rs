use std::collections::HashMap;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

#[derive(Copy, Clone, Debug)]
pub struct ADValue {
    id: usize,
    pub value: f32,
}

#[derive(Debug)]
struct PartialDiff {
    with_respect_to_id: usize,
    value: f32,
}

#[derive(Debug)]
struct TapeRecord {
    partials: Vec<PartialDiff>,
}

#[derive(Debug)]
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

#[derive(Debug)]
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

    pub fn reset(&mut self) {
        self.tape = Tape::new();
        self.gradients.clear();
    }

    pub fn create_variable(&mut self, value: f32) -> ADValue {
        let id = self.tape.len();
        self.tape.records.push(TapeRecord {
            partials: Vec::new(),
        });
        ADValue {
            id,
            value,
        }
    }

    fn compute_gradient(&mut self, y: ADValue) {
        let mut dy = vec![0.0; y.id + 1];

        dy[y.id] = 1.0;
        for i in (0..y.id+1).rev() {
            for record in &self.tape.records[i].partials {
                dy[record.with_respect_to_id] += dy[i] * record.value;
            }
        }

        self.gradients.insert(y.id, dy);
    }

    pub fn diff(&mut self, y: ADValue, wrt: ADValue) -> f32 {
        match self.gradients.get(&y.id) {
            Some(dy) => dy[wrt.id],
            None => {
                self.compute_gradient(y);
                self.gradients[&y.id][wrt.id]
            }
        }
    }

    pub fn add_many(&mut self, values: &Vec<ADValue>) -> ADValue {
        let id = self.tape.len();

        let partials = values.iter().map(|value| {
            PartialDiff {
                with_respect_to_id: value.id,
                value: 1.0,
            }
        }).collect();

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: values.iter().map(|v| v.value).sum(),
        }
    }

    pub fn add(&mut self, left: ADValue, right: ADValue) -> ADValue {
        self.add_many(&vec![left, right])
    }

    pub fn mul_many(&mut self, values: &Vec<ADValue>) -> ADValue {
        let id = self.tape.len();

        let partials = values.iter().enumerate().map(|(i, value)| {
            PartialDiff {
                with_respect_to_id: value.id,
                value: values.iter().enumerate().map(|(j, v)| {
                    if i == j {
                        1.0
                    } else {
                        v.value
                    }
                }).product(),
            }
        }).collect();

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: values.iter().map(|v| v.value).product(),
        }
    }

    pub fn mul(&mut self, left: ADValue, right: ADValue) -> ADValue {
        self.mul_many(&vec![left, right])
    }

    pub fn div(&mut self, left: ADValue, right: ADValue) -> ADValue {
        let id = self.tape.len();

        let partials = vec![
        PartialDiff {
            with_respect_to_id: left.id,
            value: 1.0 / right.value,
        },
        PartialDiff {
            with_respect_to_id: right.id,
            value: -left.value / right.value.powi(2),
        },
        ];

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: left.value / right.value,
        }
    }

    pub fn sub_many(&mut self, values: &Vec<ADValue>) -> ADValue {
        let id = self.tape.len();

        let partials = values.iter().enumerate().map(|(i, value)| {
            PartialDiff {
                with_respect_to_id: value.id,
                value: if i == 0 { 1.0 } else { -1.0 },
            }
        }).collect();

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: values.iter().enumerate().map(
                |(i, v)| if i == 0 { v.value } else { -v.value }
            ).sum(),
        }
    }

    pub fn sub(&mut self, left: ADValue, right: ADValue) -> ADValue {
        self.sub_many(&vec![left, right])
    }

    pub fn exp(&mut self, v: ADValue) -> ADValue {
        let exp = v.value.exp();

        let id = self.tape.len();

        let partials = vec![
        PartialDiff {
            with_respect_to_id: v.id,
            value: exp,
        }
        ];

        self.tape.records.push(TapeRecord {
            partials,
        });

        ADValue {
            id,
            value: exp,
        }
    }

    pub fn apply_neuron_activation(&mut self, v: ADValue, activation: &NeuronActivation) -> ADValue {
        match activation {
            NeuronActivation::LeakyReLU(alpha) => {
                let id = self.tape.len();

                let partials = vec![
                    PartialDiff {
                        with_respect_to_id: v.id,
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
        let z = ad.mul_many(&vec![x, x, y]);

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
        let exp_x_minus_y = ad.sub_many(&vec![exp_x, y]);
        let o = ad.div(y, exp_x_minus_y);

        assert_eq!(ad.diff(o, x), -0.310507656);
        assert_eq!(ad.diff(o, y), 0.077626914);
    }
}
