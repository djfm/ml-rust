use rand::prelude::*;

use crate::ml::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

pub struct LayerConfig {
    pub use_biases: bool,
    pub neurons_count: usize,
    pub neuron_activation: NeuronActivation,
    pub layer_activation: LayerActivation,
}

pub struct ScalarNetwork {
    input_size: usize,
    params: Vec<f32>,
    layers: Vec<LayerConfig>,
    error_function: ErrorFunction,
    params_offsets: Vec<usize>,
}

impl ScalarNetwork {
    pub fn new(input_size: usize, error_function: ErrorFunction, layers: Vec<LayerConfig>) -> Self {
        let mut rng = thread_rng();
        let mut params = Vec::new();
        let mut prev_size = input_size;
        let mut params_offsets = Vec::new();
        params_offsets.push(0);

        for layer in &layers {
            let size = prev_size * layer.neurons_count + if layer.use_biases { 1 } else { 0 };

            for _ in 0..size {
                params.push(rng.gen());
            }

            params_offsets.push(size);

            prev_size = layer.neurons_count;
        }

        ScalarNetwork {
            input_size,
            params,
            layers,
            error_function,
            params_offsets,
        }
    }

    pub fn weights(&self, layer: usize, neuron: usize) -> &[f32] {
        let prev_size = if layer == 0 {
            self.input_size
        } else {
            self.layers[layer - 1].neurons_count
        };

        let use_biases = if self.layers[layer].use_biases { 1 } else { 0 };
        let layer_offset = if layer == 0 { 0 } else { self.params_offsets[layer - 1] };
        let offset = layer_offset + neuron * (prev_size + use_biases);

        &self.params[offset..offset + prev_size]
    }

    pub fn bias(&self, layer: usize, neuron: usize) -> Option<f32> {
        let offset = self.params_offsets[layer] + neuron * self.layers[layer].neurons_count;

        if self.layers[layer].use_biases {
            Some(self.params[offset])
        } else {
            None
        }
    }
}
