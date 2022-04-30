use rand::prelude::*;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

use super::autodiff::{
    AutoDiff,
    ADValue,
};

pub trait Example {
    fn get_input(&self) -> Vec<f32>;
    fn get_one_hot_label(&self) -> Vec<f32>;
}

#[derive(Copy, Clone)]
struct Param {
    ad_value: ADValue,
}

struct Neuron {
    bias: Option<Param>,
    weights: Vec<Param>,
    activation: Option<NeuronActivation>,
    value: Param,
}

struct FFLayer {
    neurons: Vec<Neuron>,
    activation: Option<LayerActivation>,
}

impl FFLayer {
    fn to_vec(&self) -> Vec<ADValue> {
        self.neurons.iter().map(|n| n.value.ad_value).collect()
    }

    fn from_vec(&mut self, input: &Vec<ADValue>) {
        for (n, i) in self.neurons.iter_mut().zip(input.iter()) {
            n.value.ad_value = *i;
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

    pub fn create_variable(&mut self, value: f32) -> ADValue {
        self.autodiff.create_variable(value)
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
                for prev_neuron_pos in 0..prev_layer.neurons.len() {
                    weights.push(Param {
                        ad_value: self.create_variable(rng.gen()),
                    });
                }
            }

            let neuron = Neuron {
                weights,
                bias: if self.layers.len() == 0 {
                    None
                } else {
                    Some(Param {
                        ad_value: self.create_variable(rng.gen()),
                    })
                },
                activation: neuron_activation,
                value: Param {
                    ad_value: self.create_variable(0.0),
                },
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
        self
    }

    fn get_neuron_value(&self, layer_depth: usize, pos_in_layer: usize) -> Param {
        let neuron = &self.layers[layer_depth].neurons[pos_in_layer];
        neuron.value
    }

    pub fn feed_forward(&mut self, input: &dyn Example) -> Vec<ADValue> {
        let input_vec = input.get_input();

        if input_vec.len() != self.layers[0].neurons.len() {
            panic!("input vector length does not match the number of neurons in the first layer");
        }

        for (neuron, &input_value) in self.layers[0].neurons.iter_mut().zip(input_vec.iter()) {
            neuron.value.ad_value = self.autodiff.create_variable(input_value);
        }

        for l in 1..self.layers.len() {
            let (prev_layer, next_layers) = self.layers.split_at_mut(l);
            let layer = next_layers.first_mut().unwrap();
            let prev_layer = &prev_layer[0];

            for neuron in layer.neurons.iter_mut() {
                neuron.value.ad_value = self.autodiff.create_variable(0.0);

                for w in 0..neuron.weights.len() {
                    let contrib = self.autodiff.mul(
                        neuron.weights[w].ad_value,
                        prev_layer.neurons[w].value.ad_value,
                    );

                    neuron.value.ad_value = self.autodiff.add(
                        neuron.value.ad_value,
                        contrib,
                    );
                }

                if let Some(activation) = neuron.activation {
                    neuron.value.ad_value = self.autodiff.apply_neuron_activation(
                        neuron.value.ad_value,
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
}
