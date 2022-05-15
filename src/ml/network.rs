use rand::prelude::*;

use crate::ml::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

pub struct LayerConfig {
    neuron_activation: NeuronActivation,
    layer_activation: LayerActivation,
    params_count: usize,
    neurons_count: usize,
}

pub struct Network {
    input_size: usize,
    error_function: ErrorFunction,
    params: Vec<f32>,
    layer_configs: Vec<LayerConfig>,
    rng: ThreadRng,
}

impl Network {
    pub fn new(input_size: usize, error_function: ErrorFunction) -> Self {
        Self {
            input_size,
            error_function,
            params: vec![],
            layer_configs: vec![],
            rng: thread_rng(),
        }
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        use_biases: bool,
        neuron_activation: NeuronActivation,
        layer_activation: LayerActivation,
    ) -> &mut Self {
        let input_size = if self.layer_configs.is_empty() {
            self.input_size
        } else {
            self.layer_configs.last().expect("last layer is absent").neurons_count
        };

        let params_count = neurons_count * input_size + use_biases as usize * neurons_count;

        for _ in 0..params_count {
            self.params.push(self.rng.gen());
        }

        self.layer_configs.push(LayerConfig {
            neuron_activation,
            layer_activation,
            params_count,
            neurons_count,
        });

        self
    }
}

#[test]
fn test_create_network() {
    let mut network = Network::new(2, ErrorFunction::EuclideanDistanceSquared);
    network
        .add_layer(2, true, NeuronActivation::LeakyRelu(0.01), LayerActivation::None)
        .add_layer(1, false, NeuronActivation::None, LayerActivation::SoftMax);

    assert_eq!(network.params.len(), 8);
}
