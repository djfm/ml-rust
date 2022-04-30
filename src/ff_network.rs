use rand::prelude::*;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

use super::autodiff::{
    AutoDiff,
    ADValue,
};

pub trait ClassificationExample {
    fn get_input(&self) -> Vec<f32>;
    fn get_label(&self) -> usize;
}

#[derive(Copy, Clone)]
struct Param {
    scalar: f32,
    ad_value: ADValue,
}

struct Neuron {
    bias: Option<Param>,
    weights: Vec<Param>,
    activation: Option<NeuronActivation>,
    ad_value: ADValue,
}

struct FFLayer {
    neurons: Vec<Neuron>,
    activation: Option<LayerActivation>,
}

impl FFLayer {
    fn to_vec(&self) -> Vec<ADValue> {
        self.neurons.iter().map(|n| n.ad_value).collect()
    }

    fn from_vec(&mut self, input: &Vec<ADValue>) {
        for (n, i) in self.neurons.iter_mut().zip(input.iter()) {
            n.ad_value = *i;
        }
    }
}

pub struct Network {
    autodiff: AutoDiff,
    layers: Vec<FFLayer>,
}

impl Network {
    pub fn new() -> Network {
        Network {
            autodiff: AutoDiff::new(),
            layers: Vec::new(),
        }
    }

    pub fn autodiff(&mut self) -> &mut AutoDiff {
        &mut self.autodiff
    }

    fn create_param(&mut self, value: f32) -> Param {
        Param {
            ad_value: self.autodiff.create_variable(value),
            scalar: value,
        }
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        neuron_activation: Option<NeuronActivation>,
        layer_activation: Option<LayerActivation>,
    ) -> &mut Network {
        let mut rng = rand::thread_rng();

        let mut layer = FFLayer {
            neurons: Vec::new(),
            activation: layer_activation,
        };

        for _ in 0..neurons_count {
            let mut weights = Vec::new();

            if self.layers.len() > 0 {
                let prev_layer = self.layers.last().unwrap();
                for _ in 0..prev_layer.neurons.len() {
                    weights.push(self.create_param(rng.gen()));
                }
            }

            let neuron = Neuron {
                weights,
                bias: if self.layers.len() == 0 {
                    None
                } else {
                    Some(self.create_param(rng.gen()))
                },
                activation: neuron_activation,
                ad_value: self.autodiff.create_variable(0.0),
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
        self
    }

    pub fn feed_forward(&mut self, input: &dyn ClassificationExample) -> Vec<ADValue> {
        let input_vec = input.get_input();

        if input_vec.len() != self.layers[0].neurons.len() {
            panic!("input vector length does not match the number of neurons in the first layer");
        }

        for (neuron, &input_value) in self.layers[0].neurons.iter_mut().zip(input_vec.iter()) {
            neuron.ad_value = self.autodiff.create_variable(input_value);
        }

        for l in 1..self.layers.len() {
            let (prev_layer, next_layers) = self.layers.split_at_mut(l);
            let layer = &mut next_layers[0];
            let prev_layer = &prev_layer[0];

            for neuron in layer.neurons.iter_mut() {
                neuron.ad_value = if let Some(bias) = neuron.bias {
                    bias.ad_value
                } else {
                    self.autodiff.create_variable(0.0)
                };

                for (w, connected_neuron) in neuron.weights.iter().zip(prev_layer.neurons.iter()) {
                    let contrib = self.autodiff.mul(
                        w.ad_value,
                        connected_neuron.ad_value,
                    );

                    neuron.ad_value = self.autodiff.add(
                        neuron.ad_value,
                        contrib,
                    );
                }

                if let Some(activation) = neuron.activation {
                    neuron.ad_value = self.autodiff.apply_neuron_activation(
                        neuron.ad_value,
                        &activation,
                    );
                }
            }

            if let Some(activation) = layer.activation {
                let activated = self.autodiff.apply_layer_activation(
                    &layer.to_vec(),
                    &activation,
                );

                layer.from_vec(&activated);
            }
        }

        self.layers.last().unwrap().to_vec()
    }

    pub fn compute_example_error(&mut self, input: &dyn ClassificationExample) -> ADValue {
        let output = self.feed_forward(input);
        let mut expected = vec![self.autodiff.create_variable(0.0); output.len()];
        expected[input.get_label()] = self.autodiff.create_variable(1.0);

        self.autodiff.euclidean_distance_squared(
            &output,
            &expected,
        )
    }

    pub fn back_propagate(&mut self, error: ADValue, learning_rate: f32) {
        // Perform the actual back propagation
        for l in 1..self.layers.len() {
            let layer = &mut self.layers[l];
            for neuron in layer.neurons.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    let error_contrib = self.autodiff.diff(
                        error,
                        weight.ad_value,
                    );
                    weight.scalar -= learning_rate * error_contrib;
                }

                if let Some(mut bias) = neuron.bias {
                    let error_contrib = self.autodiff.diff(
                        error,
                        bias.ad_value,
                    );
                    bias.scalar -= learning_rate * error_contrib;
                }
            }
        }

        // Avoid explosion of the number of tracked variables
        self.autodiff.reset();

        // Prepare the network again for next automatic differentiation
        for l in 1..self.layers.len() {
            let layer = &mut self.layers[l];
            for neuron in layer.neurons.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    weight.ad_value = self.autodiff.create_variable(weight.scalar);
                }

                if let Some(mut bias) = neuron.bias {
                    bias.ad_value = self.autodiff.create_variable(bias.scalar);
                }
            }
        }
    }
}
