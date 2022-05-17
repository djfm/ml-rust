use rand::prelude::*;
use rayon::prelude::*;

use crate::ml::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
    NumberFactory,
    NumberLike,
};

pub trait ClassificationExample: Sync + Send {
    fn get_input(&self) -> &[f32];
    fn get_category(&self) -> usize;
    fn get_categories_count(&self) -> usize;
    fn get_expected_one_hot(&self) -> Vec<f32> {
        let mut expected = vec![0.0; self.get_categories_count()];
        expected[self.get_category()] = 1.0;
        expected
    }
}

pub struct Network {
    input_size: usize,
    error_function: ErrorFunction,
    params: Vec<f32>,
    layer_configs: Vec<LayerConfig>,
    rng: StdRng,
}

struct LayerConfig {
    neuron_activation: NeuronActivation,
    layer_activation: LayerActivation,
    params_count: usize,
    params_offset: usize,
    neurons_count: usize,
    use_biases: bool,
}

struct FFResult {
    error: f32,
    diffs: Vec<f32>,
    expected_category: usize,
    actual_category: usize,
    batch_size: usize,
}

impl FFResult {
    fn to_batch_result(self) -> BatchResult {
        BatchResult {
            error: self.error,
            diffs: self.diffs,
            accuracy: if self.expected_category == self.actual_category { 1.0 } else { 0.0 },
            batch_size: self.batch_size,
        }
    }
}

#[derive(Clone)]
struct BatchResult {
    error: f32,
    diffs: Vec<f32>,
    accuracy: f32,
    batch_size: usize,
}

impl BatchResult {
    fn new(batch_size: usize) -> Self {
        BatchResult {
            error: 0.0,
            diffs: vec![],
            accuracy: 0.0,
            batch_size,
        }
    }

    fn finalize(&mut self) -> &mut Self {
        self.error = 100.0 * self.error / self.batch_size as f32;
        self.accuracy = 100.0 - self.error;
        for d in &mut self.diffs {
            *d /= self.batch_size as f32;
        }
        self
    }
}

impl std::iter::Sum for BatchResult {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        if let Some(result) = iter.next() {
            let mut sum = result;

            for result in iter.skip(1) {
                sum.error += result.error;
                sum.accuracy += result.accuracy;
                for (s, a) in sum.diffs.iter_mut().zip(result.diffs.iter()) {
                    *s += *a;
                }
            }

            sum
        } else {
            BatchResult::new(0)
        }
    }
}

impl Network {
    pub fn new(input_size: usize, error_function: ErrorFunction) -> Self {
        Self {
            input_size,
            error_function,
            params: vec![],
            layer_configs: vec![],
            rng: StdRng::from_entropy(),
        }
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        use_biases: bool,
        neuron_activation: NeuronActivation,
        layer_activation: LayerActivation,
    ) -> &mut Self {
        let input_size = if self.layer_configs.is_empty() {
            self.input_size
        } else {
            self.layer_configs.last().expect("last layer is absent").neurons_count
        };

        let params_offset = if self.layer_configs.is_empty() {
            0
        } else {
            let prev_conf = self.layer_configs.last().expect("last layer is absent");
            prev_conf.params_offset + prev_conf.params_count
        };

        let params_count = neurons_count * input_size + use_biases as usize * neurons_count;

        for _ in 0..params_count {
            self.params.push(self.rng.gen());
        }

        self.layer_configs.push(LayerConfig {
            neuron_activation,
            layer_activation,
            params_count,
            params_offset,
            neurons_count,
            use_biases,
        });

        self
    }

    fn get_bias(&self, layer: usize, neuron: usize) -> f32 {
        let conf = &self.layer_configs[layer];

        if !conf.use_biases {
            return 0.0;
        }

        let prev_size = if self.layer_configs.is_empty() {
            self.input_size
        } else {
            self.layer_configs[layer - 1].neurons_count
        };
        let index = conf.params_offset + neuron * (prev_size + conf.use_biases as usize);
        self.params[index]
    }

    fn get_weights(&self, layer: usize, neuron: usize) -> &[f32] {
        let conf = &self.layer_configs[layer];
        let use_biases = conf.use_biases as usize;
        let prev_size = if self.layer_configs.is_empty() {
            self.input_size
        } else {
            self.layer_configs[layer - 1].neurons_count
        };
        let index = conf.params_offset + neuron * (prev_size + use_biases) + use_biases;
        &self.params[index .. index + prev_size]
    }

    fn feed_forward<C: ClassificationExample, N: NumberLike, F: NumberFactory<N>>(
        &self,
        nf: &mut F,
        example: &C,
    ) -> FFResult {
        let mut params: Vec<N> = Vec::with_capacity(self.params.len());

        let mut previous_activations = nf.from_scalars(example.get_input());

        for (l, conf) in self.layer_configs.iter().enumerate() {
            let activations = (0..conf.neurons_count).map(|neuron| {
                let bias = self.get_bias(l, neuron);

                if conf.use_biases {
                    params.push(nf.from_scalar(bias));
                }

                let weights = self.get_weights(l, neuron);
                let contributions = weights.iter().zip(previous_activations.iter()).map(
                    |(w, a)| {
                        let weight = nf.from_scalar(*w);
                        params.push(weight);
                        nf.mul(&weight, a)
                    }
                ).collect::<Vec<N>>();

                let mut sum = nf.from_scalar(bias);

                for c in &contributions {
                    sum = nf.add(&sum, c);
                }

                if conf.neuron_activation != NeuronActivation::None {
                    nf.activate_neuron(&sum, &conf.neuron_activation)
                } else {
                    sum
                }
            }).collect::<Vec<N>>();

            if conf.layer_activation != LayerActivation::None {
                previous_activations = nf.activate_layer(&activations, &conf.layer_activation);
            } else {
                previous_activations = activations;
            }
        }

        let expected = nf.from_scalars(&example.get_expected_one_hot());
        let error = nf.compute_error(&expected, &previous_activations, &self.error_function);

        FFResult {
            error: error.scalar(),
            diffs: params.iter().map(|p| nf.diff(&error, p)).collect(),
            expected_category: example.get_category(),
            actual_category: nf.hottest_index(&previous_activations),
            batch_size: 1,
        }
    }

    fn feed_batch_forward<C: ClassificationExample, N: NumberLike, F: NumberFactory<N>>(
        &self,
        examples: &[C],
    ) -> BatchResult {
        examples.par_iter().map(|example| {
            let mut nf = F::new();
            self.feed_forward(&mut nf, example).to_batch_result()
        }).sum::<BatchResult>()
    }
}

#[test]
fn test_create_network() {
    let mut network = Network::new(2, ErrorFunction::EuclideanDistanceSquared);
    network
        .add_layer(2, true, NeuronActivation::LeakyRelu(0.01), LayerActivation::None)
        .add_layer(1, false, NeuronActivation::None, LayerActivation::SoftMax);

    assert_eq!(network.params.len(), 8);
}
