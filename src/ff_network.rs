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
    value: f32,
    ad_value: ADValue,
}

struct Connection {
    layer_depth: usize,
    pos_in_layer: usize,
    weight: Param,
}

struct Neuron {
    bias: Option<Param>,
    connections: Vec<Connection>,
    activation: NeuronActivation,
    value: Param,
}

struct FFLayer {
    neurons: Vec<Neuron>,
    activation: LayerActivation,
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
        neuron_activation: NeuronActivation,
        layer_activation: LayerActivation,
    ) -> &mut Network {
        let mut rng = rand::thread_rng();

        let mut layer = FFLayer {
            neurons: Vec::new(),
            activation: layer_activation,
        };


        for neuron_pos in 0..neurons_count {
            let mut connections = Vec::new();

            if self.layers.len() > 0 {
                let prev_layer = self.layers.last().unwrap();
                for prev_neuron_pos in 0..prev_layer.neurons.len() {
                    let weight = rng.gen();
                    connections.push(Connection {
                        layer_depth: self.layers.len() - 1,
                        pos_in_layer: prev_neuron_pos,
                        weight: Param {
                            value: weight,
                            ad_value: self.create_variable(weight)
                        },
                    });
                }
            }

            let bias = rng.gen();
            let neuron = Neuron {
                bias: if self.layers.len() == 0 {
                    None
                } else {
                    Some(Param {
                        value: bias,
                        ad_value: self.create_variable(bias),
                    })
                },
                connections,
                activation: neuron_activation,
                value: Param {
                    value: 0.0,
                    ad_value: self.create_variable(0.0),
                },
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
        self
    }

    pub fn feed_forward(&mut self, input: &dyn Example) -> Vec<ADValue> {
        let input_vec = input.get_input();

        if input_vec.len() != self.layers[0].neurons.len() {
            panic!("input vector length does not match the number of neurons in the first layer");
        }

        for (neuron_pos, neuron) in self.layers[0].neurons.iter_mut().enumerate() {
            neuron.value.value = input_vec[neuron_pos];
            neuron.value.ad_value = self.autodiff.create_variable(neuron.value.value);
        }

        for l in 1..self.layers.len() {
            for n in 0..self.layers[l].neurons.len() {
                let neuron = &mut self.layers[l].neurons[n];
                let bias = match neuron.bias {
                    Some(bias) => bias.value,
                    None => 0.0,
                };

                neuron.value = Param{
                    value: bias,
                    ad_value: self.autodiff.create_variable(bias),
                };

                for c in 0..self.layers[l].neurons[n].connections.len() {
                    let connection = &mut self.layers[l].neurons[n].connections[c];
                    let connection_depth = connection.layer_depth;
                    let connection_pos = connection.pos_in_layer;
                    let connected_layer = &self.layers[connection_depth];
                    let connected_value = connected_layer.neurons[connection_pos].value;
                    let connected_weight = self.layers[l].neurons[n].connections[c].weight;

                    let contrib = self.autodiff.mul(
                        connected_value.ad_value,
                        connected_weight.ad_value,
                    );

                    let sum = self.autodiff.add(
                        self.layers[l].neurons[n].value.ad_value,
                        contrib,
                    );
                    self.layers[l].neurons[n].value.ad_value = sum;
                    self.layers[l].neurons[n].value.value = sum.scalar();
                }
            }
        }

        self.layers.last().unwrap().neurons.iter().map(|n| n.value.ad_value).collect()
    }
}
