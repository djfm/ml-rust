use rand::prelude::*;

use crate::ml::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
    ClassificationExample,
    NumberFactory,
    FloatFactory,
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
            let size = (prev_size + if layer.use_biases { 1 } else { 0 }) * layer.neurons_count ;

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

    pub fn params(&mut self) -> &mut Vec<f32> {
        &mut self.params
    }

    pub fn params_count(&self) -> usize {
        self.params.len()
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

    pub fn bias_value(&self, layer: usize, neuron: usize) -> f32 {
        match self.bias(layer, neuron) {
            Some(bias) => bias,
            None => 0.0,
        }
    }

    pub fn layers_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_config(&self, layer: usize) -> &LayerConfig {
        &self.layers[layer]
    }

    pub fn error_function(&self) -> ErrorFunction {
        self.error_function
    }

    pub fn predict(&self, input: &dyn ClassificationExample) -> usize {
        let mut ff = FloatFactory::new();

        let mut previous_activations = input.get_input();

        for l in 0..self.layers.len() {
            let config = &self.layers[l];
            let activations = (0..config.neurons_count).into_iter().map(
                |n| {
                    let a = self.bias_value(l, n) +
                    self.weights(l, n).iter().zip(
                        previous_activations.iter()
                    ).fold(0.0, |acc, (w, x)| acc + w * x);

                    ff.activate_neuron(a, &config.neuron_activation)
                }
            ).collect::<Vec<_>>();

            previous_activations = if config.layer_activation == LayerActivation::None {
                activations
            } else {
                ff.activate_layer(&activations, &config.layer_activation)
            }
        }

        ff.get_max_value(&previous_activations)
    }

    pub fn compute_batch_accuracy<T: ClassificationExample>(&self, examples: &[T]) -> f32 {
        let mut correct = 0;
        let total = examples.len();

        for example in examples {
            let prediction = self.predict(example);
            if prediction == example.get_label() {
                correct += 1;
            }
        }

        100.0 * correct as f32 / total as f32
    }
}
