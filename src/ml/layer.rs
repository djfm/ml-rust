use crate::ml::{
    math::{
        NumberLike,
        NumberFactory,
        LayerActivation,
        CellActivation,
    },
};

pub struct Layer<
    CellT: NumberLike<Factory>,
    Factory: NumberFactory<CellT>,
> {
    pub weights: Vec<CellT>,
    pub biases: Vec<CellT>,
    pub config: LayerConfig,
    phantom: std::marker::PhantomData<Factory>,
}

#[derive(Copy, Clone)]
pub struct LayerConfig {
    pub layer_size: usize,
    pub input_size: usize,
    pub layer_activation: LayerActivation,
    pub cell_activation: CellActivation,
    pub has_biases: bool,
}

impl <
    CellT: NumberLike<FactoryT>,
    FactoryT: NumberFactory<CellT>,
> Layer<CellT, FactoryT> {
    pub fn new(
        config: &LayerConfig,
        input_size: usize,
    ) -> Self {
        let weights = (0..config.layer_size*input_size).map(
            |_| FactoryT::small_rand()
        ).collect();

        let biases = if config.has_biases {
            vec![
                FactoryT::zero();
                config.layer_size
            ]
        } else {
            vec![]
        };

        Layer {
            weights, biases,
            config: LayerConfig {
                input_size,
                ..*config
            },
            phantom: std::marker::PhantomData,
        }
    }

    pub fn weights_for_cell(&self, neuron_id: usize) -> &[CellT] {
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
