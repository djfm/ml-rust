use crate::ml::{
    Layer,
    LayerConfig,
    NumberLike,
    NumberFactory,
    ErrorFunction,
};

pub struct Network<N> where N: NumberLike {
    layers: Vec<Layer<N>>,
    error_function: ErrorFunction,
}

impl <N> Network<N> where N: NumberLike {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            error_function: ErrorFunction::EuclideanDistanceSquared
        }
    }

    pub fn add_layer<F: NumberFactory<N>>(
        &mut self,
        nf: &mut F,
        config: LayerConfig
    ) -> &mut Self {
        let prev_size = if self.layers.is_empty() {
            0
        } else {
            self.layers.last().expect("should be a first layer").config().neurons_count
        };

        let layer = Layer::new(nf, config, prev_size);
        self.layers.push(layer);
        self
    }
}
