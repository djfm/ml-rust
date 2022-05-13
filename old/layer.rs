use crate::ml::{
    NumberLike,
    NumberFactory,
    NeuronActivation,
    LayerActivation,
};

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
