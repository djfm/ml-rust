use crate::ml::{
    AD,
    ADNumber,
    Layer,
    LayerActivation,
    ErrorFunction,
    LayerConfig,
    math::{
        one_hot_label,
    }
};

pub struct Network<'a>
{
    nf: AD,
    layers: Vec<Layer<'a>>,
    error_function: ErrorFunction,
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

pub trait TrainingSample<'a>
{
    fn get_input(&'a self, nf: &'a AD) -> Vec<ADNumber<'a>>;
    fn get_label(&self) -> usize;
    fn get_expected_one_hot(&'a self, nf: &'a AD) -> Vec<ADNumber<'a>>;
}

impl <'a> Network<'a>
{
    pub fn new(nf: AD) -> Self {
        Network {
            nf,
            layers: Vec::new(),
            error_function: ErrorFunction::EuclideanDistanceSquared,
        }
    }

    pub fn nf(&self) -> &AD {
        &self.nf
    }

    pub fn add_layer(
        &'a mut self,
        config: LayerConfig,
    ) -> &mut Self {
        if self.layers.is_empty() && config.input_size == 0 {
            panic!("Input size must be specified for the first layer");
        }

        let input_size = if self.layers.is_empty() {
            config.input_size
        } else {
            self.layers.last().expect("there should be an input layer").config.input_size
        };

        let nf = &self.nf;
        let weights = (0..config.layer_size*input_size).into_iter().map(
            |_| nf.create_random_variable()
        ).collect::<Vec<ADNumber<'a>>>();

        let biases = if config.has_biases {
            vec![
                self.nf.create_constant(0.0);
                config.layer_size
            ]
        } else {
            vec![]
        };

        let layer = Layer {
            weights, biases,
            config: LayerConfig {
                input_size,
                ..config.clone()
            },
        };

        self.layers.push(layer);

        self
    }

    pub fn set_error_function(
        &mut self,
        error_function: ErrorFunction,
    ) -> &mut Self {
        self.error_function = error_function;
        self
    }

    pub fn feed_forward(&'a self, input: &'a dyn TrainingSample<'a>) -> Vec<ADNumber<'a>> {
        let mut previous_activations = input.get_input(&self.nf);

        for layer in self.layers.iter().skip(1) {
            let n_cells = layer.config.layer_size;
            let mut layer_activations = if layer.config.has_biases {
                layer.biases.clone()
            } else {
                vec![self.nf.create_constant(0.0); n_cells]
            };

            for cell_id in 0..n_cells {
                let cell = layer
                    .weights_for_cell(cell_id).iter().zip(
                        previous_activations.iter()
                    ).map(
                        |(weight, activation)| *weight * *activation
                    ).reduce(
                        |acc, x| acc + x
                    ).expect("summation failed");

                let activated = layer.config.cell_activation.compute(&cell);

                layer_activations[cell_id] += activated;

            }

            if layer.config.layer_activation != LayerActivation::None {
                previous_activations = layer.config.layer_activation.compute(
                    &layer_activations
                );
            } else {
                previous_activations = layer_activations;
            }

        }

        previous_activations
    }

    pub fn predict(&'a self, input: &'a dyn TrainingSample<'a>) -> usize {
        one_hot_label(&self.feed_forward(input))
    }

    pub fn compute_sample_error(&'a self, sample: &'a dyn TrainingSample<'a>) -> ADNumber<'a> {
        let error_function = self.error_function;
        let actual = self.feed_forward(sample);
        let expected = sample.get_expected_one_hot(&self.nf);
        error_function.compute(
            &expected,
            &actual
        )
    }

    pub fn compute_batch_error<
        S: TrainingSample<'a>
    >(
        &'a self,
        samples: &'a [S]
    ) -> ADNumber<'a> {
        let batch_size = self.nf.create_constant(samples.len() as f32);

        let mut error = self.nf.create_constant(0.0);

        for s in samples.iter() {
            error += self.compute_sample_error(s);
        }

        error / batch_size
    }

    pub fn compute_accuracy<
            S: TrainingSample<'a>
    >(&'a mut self, samples: &'a [S]) -> f32 {
        let mut correct = 0;

        for s in samples {
            if self.predict(s) == s.get_label() {
                correct += 1;
            }
        }

        100.0 * correct as f32 / samples.len() as f32
    }

    pub fn back_propagate(&'a mut self, error: &ADNumber<'a>, tconf: &TrainingConfig) -> &'a mut Self {
        for layer in self.layers.iter_mut() {
            for weight in layer.weights.iter_mut() {
                let delta = error.diff(weight) * tconf.learning_rate;
                *weight -= self.nf.create_constant(delta);
            }

            if layer.config.has_biases {
                for bias in layer.biases.iter_mut() {
                    let delta = error.diff(bias) * tconf.learning_rate;
                    *bias -= self.nf.create_constant(delta);
                }
            }
        }

        self
    }
}
