use rand::prelude::*;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

struct Param {
    value: f32,
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
}

struct FFLayer {
    neurons: Vec<Neuron>,
    activation: LayerActivation,
}

pub struct FFNetwork {
    layers: Vec<FFLayer>,
}

impl FFNetwork {
    pub fn new() -> FFNetwork {
        FFNetwork {
            layers: Vec::new(),
        }
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        neuron_activation: NeuronActivation,
        layer_activation: LayerActivation,
    ) -> &mut FFNetwork {
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
                    connections.push(Connection {
                        layer_depth: self.layers.len() - 1,
                        pos_in_layer: prev_neuron_pos,
                        weight: Param { value: rng.gen() },
                    });
                }
            }

            let neuron = Neuron {
                bias: if self.layers.len() == 0 {
                    None
                } else {
                    Some(Param { value: rng.gen() })
                },
                connections,
                activation: neuron_activation,
            };
            layer.neurons[neuron_pos] = neuron;
        }
        self.layers.push(layer);
        self
    }
}
