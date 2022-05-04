use crate::ml::math::{
    NumberFactory,
    NumberLike,
    LayerActivation,
    CellActivation,
    ErrorFunction,
};

pub trait TrainingSample<T, F>
where
    T: NumberLike<F>,
    F: NumberFactory<T>,
{
    fn get_input(&self) -> Vec<T>;
    fn get_label(&self) -> usize;
    fn get_expected_one_hot(&self) -> Vec<T>;
}

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

    fn weights_for_cell(&self, neuron_id: usize) -> &[CellT] {
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

    pub fn feed_forward(&mut self, input: &dyn TrainingSample<CellT, FactoryT>) -> &Vec<CellT> {
        for (c, i) in self.layers[0].cells.iter_mut().zip(input.get_input()) {
            *c = i;
        }

        for l in 1..self.layers.len() {
            let (prev_layers, nex_layers) = self.layers.split_at_mut(l);
            let prev_layer = &prev_layers[l-1];
            let layer = &mut nex_layers[l];

            for c in 0..layer.cells.len() {
                let mut sum = layer.biases[c];
                for (w, prev_cell) in layer.weights_for_cell(c).iter().zip(prev_layer.cells.iter()) {
                    sum += *w * *prev_cell;
                }
                layer.cells[c] = layer.config.cell_activation.compute(&sum);
            }

            if layer.config.layer_activation != LayerActivation::None {
                layer.cells = layer.config.layer_activation.compute(&layer.cells);
            }
        }

        &self.layers.last().unwrap().cells
    }

    pub fn compute_sample_error(&mut self, sample: &dyn TrainingSample<CellT, FactoryT>) -> CellT {
        let error_function = self.error_function;
        let actual = self.feed_forward(sample);
        let expected = sample.get_expected_one_hot();
        error_function.compute(
            &expected,
            &actual
        )
    }

    pub fn compute_batch_error<
        S: TrainingSample<CellT, FactoryT>
    >(
        &mut self,
        samples: &[S]
    ) -> CellT {
        samples.iter().map(
            |s| self.compute_sample_error(s)
        ).reduce(
            |a, b| a + b
        ).expect("Cannot compute batch error")
    }
}
