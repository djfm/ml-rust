use crate::ml::{
    NumberLike,
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

pub trait NumberFactory<N> where N: NumberLike {
    fn create_variable(&mut self, scalar: f32) -> N;
    fn create_random_variable(&mut self) -> N;

    fn multiply(&mut self, left: N, right: N) -> N;
    fn divide(&mut self, left: N, right: N) -> N;
    fn addition(&mut self, left: N, right: N) -> N;
    fn subtract(&mut self, left: N, right: N) -> N;

    fn exp(&mut self, operand: N) -> N;

    fn to_scalar(&self, operand: N) -> f32;

    fn activate_neuron(&mut self, neuron: N, activation: &NeuronActivation) -> N {
        match activation {
            NeuronActivation::None => self.unary_operation(neuron, self.to_scalar(neuron), 1.0),
            NeuronActivation::ReLu => {
                if neuron > self.create_variable(0.0) {
                    self.unary_operation(neuron, self.to_scalar(neuron), 1.0)
                } else {
                    self.create_variable(0.0)
                }
            },
            NeuronActivation::LeakyReLU(alpha) => {
                if neuron > self.create_variable(0.0) {
                    self.unary_operation(neuron, self.to_scalar(neuron), 1.0)
                } else {
                    self.unary_operation(neuron, self.to_scalar(neuron), *alpha)
                }
            }
        }
    }

    fn activate_layer(&mut self, layer: &Vec<N>, activation: &LayerActivation) -> Vec<N> {
        match activation {
            LayerActivation::None => layer.clone(),
            LayerActivation::SoftMax => {
                let mut res = vec![self.create_variable(0.0); layer.len()];

                let mut sum = self.create_variable(0.0);
                for (i, v) in layer.iter().enumerate() {
                    let exp = self.exp(*v);
                    res[i] = exp;
                    sum = self.addition(sum, exp);
                }

                for v in res.iter_mut() {
                    *v = self.divide(*v, sum);
                }

                res
            }
        }
    }

    fn compute_error(&mut self, expected: &Vec<N>, actual: &Vec<N>, error_function: &ErrorFunction) -> N {
        match error_function {
            ErrorFunction::EuclideanDistanceSquared => {
                let mut sum = self.create_variable(0.0);

                for (e, a) in expected.iter().zip(actual.iter()) {
                    let diff = self.subtract(*e, *a);
                    let square = self.multiply(diff, diff);
                    sum = self.addition(sum, square);
                }

                sum
            }
        }
    }

    fn get_label(&mut self, vec: &Vec<N>) -> usize {
        let mut max = self.create_variable(0.0);
        let mut index = 0;

        for (i, v) in vec.iter().enumerate() {
            if *v > max {
                max = *v;
                index = i;
            }
        }

        index
    }

    fn has_automatic_diff(&self) -> bool;

    fn binary_operation(
        &mut self,
        left: N, right: N,
        result: f32,
        diff_left: f32, diff_right: f32,
    ) -> N;

    fn unary_operation(
        &mut self,
        operand: N,
        result: f32,
        diff: f32,
    ) -> N;
}

pub trait DifferentiableNumberFactory<N>: NumberFactory<N> where N: NumberLike {
    fn diff(&mut self, y: N, x: N) -> f32;
}
