use crate::ml::math::{
    NumberFactory,
    NumberLike,
    LayerActivation,
    SingleActivation,
};

pub struct Layer<
    CellT: NumberLike<Factory>,
    Factory: NumberFactory<CellT>,
    const n_cells: usize,
    const prev_layer_size: usize
> {
    weights: Vec<CellT>,
    biases: Vec<CellT>,
    cells: Vec<CellT>,
    config: layer::Config,
    phantom: std::marker::PhantomData<Factory>,
}

impl <
    CellT: NumberLike<FactoryT>,
    FactoryT: NumberFactory<CellT>,
    const n_cells: usize,
    const prev_layer_size: usize,
> Layer<CellT, FactoryT, n_cells, prev_layer_size> {
    pub fn new(config: &layer::Config) -> Self {
        let weights = (0..n_cells*prev_layer_size).map(
            |_| FactoryT::small_rand()
        ).collect();

        let biases = vec![
            FactoryT::zero();
            n_cells
        ];
        let cells = vec![
            FactoryT::zero();
            n_cells
        ];
        Layer {
            weights,
            biases,
            cells,
            config: *config,
            phantom: std::marker::PhantomData,
        }
    }
}

mod layer {
    use super::*;

    #[derive(Copy, Clone)]
    pub struct Config {
        pub input_size: usize,
        pub n_cells_per_input: usize,
        pub layer_depth: usize,
        pub layer_activation: LayerActivation,
        pub cell_activation: SingleActivation,
    }
}

pub struct Network<
    Factory, CellT,
    const input_size: usize,
    const default_input_size: usize,
> where
        CellT: NumberLike<Factory>,
        Factory: NumberFactory<CellT>,
{
    layers: Vec<Layer<CellT, Factory, input_size, default_input_size>>,
    phantom: std::marker::PhantomData<Factory>,
}
