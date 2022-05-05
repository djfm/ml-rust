use crate::ml::{
    Layer,
    LayerActivation,
    LayerConfig,
    math::{
        one_hot_label,
        Differentiable,
        ErrorFunction,
        NumberFactory,
        NumberLike,
    },
};

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

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
}

impl TrainingConfig {
    pub fn new() -> Self {
        TrainingConfig {
            learning_rate: 0.01,
            batch_size: 32,
            epochs: 5,
        }
    }
}

pub trait TrainingSample<T, F>
where
    T: NumberLike<F>,
    F: NumberFactory<T>,
{
    fn get_input(&self) -> Vec<T>;
    fn get_label(&self) -> usize;
    fn get_expected_one_hot(&self) -> Vec<T>;
}

impl <CellT, FactoryT> Network<CellT, FactoryT>
where
    CellT: NumberLike<FactoryT>,
    FactoryT: NumberFactory<CellT>,
{
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            error_function: ErrorFunction::EuclideanDistanceSquared,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn add_layer(
        &mut self,
        config: LayerConfig,
    ) -> &mut Self {
        if self.layers.is_empty() && config.input_size == 0 {
            panic!("Input size must be specified for the first layer");
        }

        self.layers.push(
            Layer::new(
                &config,
                if self.layers.len() > 0 {
                    self.layers.last().unwrap().config.layer_size
                } else {
                    config.input_size
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

    pub fn feed_forward(&self, input: &dyn TrainingSample<CellT, FactoryT>) -> Vec<CellT> {
        let mut previous_activations = input.get_input();

        for layer in self.layers.iter().skip(1) {
            let n_cells = layer.config.layer_size;
            let mut layer_activations = if layer.config.has_biases {
                layer.biases.clone()
            } else {
                vec![FactoryT::zero(); n_cells]
            };

            for cell_id in 0..n_cells {
                layer_activations[cell_id] += layer
                    .weights_for_cell(cell_id).iter().zip(
                        previous_activations.iter()
                    ).map(
                        |(weight, activation)| *weight * *activation
                    ).reduce(
                        |acc, x| acc + x
                    ).expect("summation failed");

                layer_activations[cell_id] = layer.config.cell_activation.compute(
                    &layer_activations[cell_id]
                );
            }

            if layer.config.layer_activation != LayerActivation::None {
                layer_activations = layer.config.layer_activation.compute(
                    &layer_activations
                );
            }

            previous_activations = layer_activations;
        }

        previous_activations
    }

    pub fn predict(&mut self, input: &dyn TrainingSample<CellT, FactoryT>) -> usize {
        one_hot_label(&self.feed_forward(input))
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
        let batch_size = FactoryT::from_scalar(samples.len() as f32);

        samples.iter().map(
            |s| self.compute_sample_error(s)
        ).reduce(
            |a, b| a + b
        ).expect("Cannot compute batch error") / batch_size
    }

    pub fn compute_accuracy<S: TrainingSample<CellT, FactoryT>>(&mut self, samples: &[S]) -> f32 {
        let mut correct = 0;

        for s in samples {
            if self.predict(s) == s.get_label() {
                correct += 1;
            }
        }

        100.0 * correct as f32 / samples.len() as f32
    }
}

impl <'a, N: NumberLike<F> + Differentiable<N, F>, F: NumberFactory<N>> Network<N, F> {
    pub fn back_propagate(&mut self, error: &N, tconf: &TrainingConfig) -> &mut Self {
        for layer in self.layers.iter_mut() {
            for weight in layer.weights.iter_mut() {
                let delta = error.diff(weight) * tconf.learning_rate;
                *weight -= F::from_scalar(delta);
            }

            if layer.config.has_biases {
                for bias in layer.biases.iter_mut() {
                    let delta = error.diff(bias) * tconf.learning_rate;
                    *bias -= F::from_scalar(delta);
                }
            }
        }

        self
    }
}
