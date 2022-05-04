use crate::ml::math::{
    NumberFactory,
    NumberLike,
    LayerActivation,
    CellActivation,
    ErrorComputer,
    ErrorFunction,
};

pub struct Layer<
    CellT: NumberLike<Factory>,
    Factory: NumberFactory<CellT>,
> {
    weights: Vec<CellT>,
    biases: Vec<CellT>,
    cells: Vec<CellT>,
    config: LayerConfig,
    phantom: std::marker::PhantomData<Factory>,
}

impl <
    CellT: NumberLike<FactoryT>,
    FactoryT: NumberFactory<CellT>,
> Layer<CellT, FactoryT> {
    fn new(
        config: &LayerConfig,
        prev_layer_size: usize,
    ) -> Self {
        let weights = (0..config.layer_size*prev_layer_size).map(
            |_| FactoryT::small_rand()
        ).collect();

        let biases = vec![
            FactoryT::zero();
            config.layer_size
        ];
        let cells = vec![
            FactoryT::zero();
            config.layer_size
        ];
        Layer {
            weights,
            biases,
            cells,
            config: LayerConfig {
                prev_layer_size,
                ..*config
            },
            phantom: std::marker::PhantomData,
        }
    }

    fn weights_for_neuron(&self, neuron_id: usize) -> &[CellT] {
        let weights_per_neuron = self.config.prev_layer_size;
        &self.weights[
            neuron_id*weights_per_neuron..(neuron_id+1)*weights_per_neuron
        ]
    }
}


#[derive(Copy, Clone)]
pub struct LayerConfig {
    pub layer_size: usize,
    pub prev_layer_size: usize,
    pub layer_activation: LayerActivation,
    pub cell_activation: CellActivation,
}

impl LayerConfig {
    pub fn new(layer_size: usize) -> Self {
        LayerConfig {
            layer_size,
            prev_layer_size: 0,
            layer_activation: LayerActivation::None,
            cell_activation: CellActivation::LeakyReLU(0.01),
        }
    }
}

pub struct Network<
    CellT, FactoryT
> where
        FactoryT: NumberFactory<CellT>,
        CellT: NumberLike<FactoryT>,
{
    layers: Vec<Layer<CellT, FactoryT>>,
    error_function: ErrorFunction,
    phantom: std::marker::PhantomData<FactoryT>,
}

impl <CellT, FactoryT> Network<CellT, FactoryT>
where
    CellT: NumberLike<FactoryT>,
    FactoryT: NumberFactory<CellT>,
{
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            error_function: ErrorFunction::None,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn add_layer(
        &mut self,
        config: LayerConfig,
    ) -> &mut Self {
        self.layers.push(
            Layer::new(
                &config,
                if self.layers.len() > 0 {
                    self.layers.last().unwrap().cells.len()
                } else {
                    0
                }
            )
        );

        self
    }

    pub fn set_error_function(
        &mut self,
        error_function: ErrorFunction,
    ) -> &mut Self {
        self.error_function = error_function;
        self
    }
}
