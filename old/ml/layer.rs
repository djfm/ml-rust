use crate::ml::{
    AD, ADNumber,
    math::{
        LayerActivation,
        CellActivation,
    },
    Network,
};

pub struct Layer<'a> {
    pub weights: Vec<ADNumber<'a>>,
    pub biases: Vec<ADNumber<'a>>,
    pub config: LayerConfig,
}

#[derive(Copy, Clone)]
pub struct LayerConfig {
    pub layer_size: usize,
    pub input_size: usize,
    pub layer_activation: LayerActivation,
    pub cell_activation: CellActivation,
    pub has_biases: bool,
}

impl <'a> Layer<'a> {
    pub fn new(
        network: &'a Network,
        config: &LayerConfig,
        input_size: usize,
    ) -> Self {
        let nf = network.nf();

        let weights = (0..config.layer_size*input_size).map(
            |_| nf.create_random_variable()
        ).collect();

        let biases = if config.has_biases {
            vec![
                nf.create_constant(0.0);
                config.layer_size
            ]
        } else {
            vec![]
        };

        Self {
            weights, biases,
            config: LayerConfig {
                input_size,
                ..*config
            },
        }
    }

    pub fn weights_for_cell(&self, neuron_id: usize) -> &[ADNumber] {
        let weights_per_neuron = self.config.input_size;
        &self.weights[
            neuron_id*weights_per_neuron..(neuron_id+1)*weights_per_neuron
        ]
    }
}

impl LayerConfig {
    pub fn new(layer_size: usize) -> Self {
        LayerConfig {
            layer_size,
            input_size: 0,
            layer_activation: LayerActivation::None,
            cell_activation: CellActivation::LeakyReLU(0.01),
            has_biases: true,
        }
    }
}
