use crate::ml::{
    NumberLike,
    NumberFactory,
    NeuronActivation,
    LayerActivation,
};

#[derive(Copy, Clone, Debug)]
pub struct LayerConfig {
    pub neurons_count: usize,
    pub neuron_activation: NeuronActivation,
    pub layer_activation: LayerActivation,
    pub use_biases: bool,
}

impl LayerConfig {
    pub fn new(neurons_count: usize) -> Self {
        LayerConfig {
            neurons_count,
            neuron_activation: NeuronActivation::LeakyReLU(0.01),
            layer_activation: LayerActivation::None,
            use_biases: true,
        }
    }
}

pub struct Layer<N> where N: NumberLike {
    config: LayerConfig,
    weights: Vec<N>,
    biases: Vec<N>,
}

impl <N> Layer<N> where N: NumberLike {
    pub fn new<F: NumberFactory<N>>(
        nf: &mut F,
        config: LayerConfig,
        prev_size: usize,
    ) -> Self {
        let weights = if prev_size == 0 {
            Vec::new()
        } else {
            (0..prev_size*config.neurons_count).map(|_| nf.create_random_variable()).collect()
        };

        let biases = if config.use_biases {
            (0..config.neurons_count).map(|_| nf.create_random_variable()).collect()
        } else {
            Vec::new()
        };

        Self {
            config,
            weights,
            biases
        }
    }

    pub fn config(&self) -> &LayerConfig {
        &self.config
    }
}
