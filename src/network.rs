use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    ErrorFunction,
    LayerActivation,
    NeuronActivation,
    NumberFactory,
    NumberLike,
    TrainingConfig,
};

pub trait ClassificationExample: Sync + Send + Clone {
    fn get_input(&self) -> Vec<f32>;
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
}

struct LayerConfig {
    neuron_activation: NeuronActivation,
    layer_activation: LayerActivation,
    params_count: usize,
    params_offset: usize,
    neurons_count: usize,
    use_biases: bool,
    drop_out: f32,
}

pub struct FFResult {
    error: f32,
    diffs: Vec<f32>,
    expected_category: usize,
    actual_category: usize,
}

impl FFResult {
    pub fn new() {
        Default::default()
    }

    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn diffs(&self) -> &[f32] {
        &self.diffs
    }

    pub fn expected_category(&self) -> usize {
        self.expected_category
    }

    pub fn actual_category(&self) -> usize {
        self.actual_category
    }
}

impl FFResult {
    fn to_batch_result(self) -> BatchResult {
        BatchResult {
            error: self.error,
            diffs: self.diffs,
            accuracy: if self.expected_category == self.actual_category {
                1.0
            } else {
                0.0
            },
            batch_size: 1,
        }
    }
}

#[derive(Clone)]
pub struct BatchResult {
    error: f32,
    diffs: Vec<f32>,
    accuracy: f32,
    batch_size: usize,
}

impl BatchResult {
    pub fn error(&self) -> f32 {
        self.error
    }

    pub fn accuracy(&self) -> f32 {
        self.accuracy
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn diffs(&self) -> &[f32] {
        &self.diffs
    }

    pub fn aggregate(results: &[BatchResult]) -> BatchResult {
        let mut sum = BatchResult {
            error: 0.0,
            diffs: vec![0.0; results[0].diffs.len()],
            accuracy: 0.0,
            batch_size: 0,
        };

        for result in results.iter() {
            sum.error += result.error;
            sum.accuracy += result.accuracy;
            sum.batch_size += result.batch_size;
            for (i, diff) in result.diffs.iter().enumerate() {

                if !diff.is_nan() && !diff.is_infinite() {
                    sum.diffs[i] += diff;
                }

                if sum.diffs[i].is_nan() {
                    panic!("sum of diffs for param {} is NaN", i);
                }
            }
        }

        sum.error /= sum.batch_size as f32 / 100.0;
        sum.accuracy /= sum.batch_size as f32 / 100.0;

        sum
    }
}

impl Network {
    pub fn new(input_size: usize, error_function: ErrorFunction) -> Self {
        Self {
            input_size,
            error_function,
            params: vec![],
            layer_configs: vec![],
        }
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        use_biases: bool,
        drop_out: f32,
        neuron_activation: NeuronActivation,
        layer_activation: LayerActivation,
    ) -> &mut Self {
        let is_first_layer = self.layer_configs.is_empty();

        let input_size = if is_first_layer {
            self.input_size
        } else {
            self.layer_configs
                .last()
                .expect("this is not the first layer")
                .neurons_count
        };

        let params_offset = if is_first_layer {
            0
        } else {
            let prev_conf = self
                .layer_configs
                .last()
                .expect("this is not the first layer");
            prev_conf.params_offset + prev_conf.params_count
        };

        let params_count = neurons_count * (input_size + use_biases as usize);

        for _ in 0..params_count {
            let rnd: f32 = thread_rng().gen();
            self.params.push(rnd / params_count as f32 / 100.0);
        }

        self.layer_configs.push(LayerConfig {
            neuron_activation,
            layer_activation,
            params_count,
            params_offset,
            neurons_count,
            use_biases,
            drop_out,
        });

        self
    }

    fn get_bias(&self, layer: usize, neuron: usize) -> f32 {
        let conf = self.layer_configs.get(layer).expect("valid layer index");

        if !conf.use_biases {
            return 0.0;
        }

        let prev_size = if layer == 0 {
            self.input_size
        } else {
            self.layer_configs[layer - 1].neurons_count
        };

        let index = conf.params_offset + neuron * (prev_size + conf.use_biases as usize);
        self.params[index]
    }

    fn get_weights_range(&self, layer: usize, neuron: usize) -> (usize, usize) {
        let conf = self.layer_configs.get(layer).expect("valid layer index");
        let use_biases = conf.use_biases as usize;
        let prev_size = if layer == 0 {
            self.input_size
        } else {
            self.layer_configs[layer - 1].neurons_count
        };
        let index = conf.params_offset + neuron * (prev_size + use_biases) + use_biases;
        (index, index + prev_size)
    }

    fn get_weights(&self, layer: usize, neuron: usize) -> &[f32] {
        let (start, end) = self.get_weights_range(layer, neuron);
        &self.params[start..end]
    }

    pub fn feed_forward<C: ClassificationExample, N: NumberLike, F: NumberFactory<N>>(
        &self,
        nf: &mut F,
        example: &C,
        predict_mode: bool,
    ) -> FFResult {
        let mut params: Vec<N> = Vec::with_capacity(self.params.len());

        let mut previous_activations = nf.constants(&example.get_input());

        for (l, conf) in self.layer_configs.iter().enumerate() {
            let activations = (0..conf.neurons_count)
                .map(|neuron| {
                    let use_param = || predict_mode || thread_rng().gen::<f32>() >= conf.drop_out;

                    let bias = self.get_bias(l, neuron);

                    let mut sum = if let Some(dnf) = nf.get_as_differentiable() {
                        if conf.use_biases {
                            let var = if use_param() { dnf.variable(bias) } else { dnf.constant(0.0) };
                            params.push(var);
                            var
                        } else {
                            dnf.constant(bias)
                        }
                    } else {
                        nf.constant(bias)
                    };

                    let weights = self.get_weights(l, neuron);
                    let contributions = weights
                        .iter()
                        .zip(previous_activations.iter())
                        .map(|(&w, &a)| {
                            let weight = if predict_mode {
                                nf.constant(w * (1.0 - conf.drop_out))
                            } else if let Some(dnf) = nf.get_as_differentiable() {
                                let w = if use_param() { dnf.variable(w) } else { dnf.constant(0.0) };
                                params.push(w);
                                w
                            } else {
                                nf.constant(w)
                            };

                            nf.mul(weight, a)

                        })
                        .collect::<Vec<N>>();

                    for &c in &contributions {
                        sum = nf.add(sum, c);
                    }

                    sum = if conf.neuron_activation != NeuronActivation::None {
                        nf.activate_neuron(&sum, &conf.neuron_activation)
                    } else {
                        sum
                    };

                    sum

                })
                .collect::<Vec<N>>();

            if conf.layer_activation != LayerActivation::None {
                previous_activations = nf.activate_layer(&activations, &conf.layer_activation);
            } else {
                previous_activations = activations;
            }
        }

        let expected = nf.constants(&example.get_expected_one_hot());
        let error = nf.compute_error(&expected, &previous_activations, &self.error_function);

        let diffs = match nf.get_as_differentiable() {
            Some(dnf) => if predict_mode { vec![] } else {
                params.iter().map(|p| dnf.diff(&error, p)).collect()
            },
            None => vec![],
        };

        FFResult {
            error: error.scalar(),
            diffs,
            expected_category: example.get_category(),
            actual_category: nf.hottest_index(&previous_activations),
        }
    }

    pub fn feed_batch_forward<
        C: ClassificationExample,
        N: NumberLike,
        NumberFactoryCreatorFunction,
        F: NumberFactory<N>,
    >(
        &self,
        cnf: NumberFactoryCreatorFunction,
        examples: &[C],
        predict_mode: bool,
    ) -> BatchResult
    where
        NumberFactoryCreatorFunction: Fn() -> F + Sync,
    {
        let results: Vec<BatchResult> = examples
            .par_iter()
            .map(|example| {
                let mut nf = cnf();
                self.feed_forward(&mut nf, example, predict_mode)
                    .to_batch_result()
            })
            .collect();

        BatchResult::aggregate(&results)
    }

    pub fn back_propagate(&mut self, diffs: &[f32], t_conf: &TrainingConfig) -> &mut Self {
        if self.params.len() != diffs.len() {
            panic!("params and diffs have different lengths");
        }

        for (p, d) in self.params.iter_mut().zip(diffs.iter()) {
            *p -= t_conf.learning_rate() * *d;
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AutoDiff, FloatFactory};

    #[derive(Clone)]
    struct TestExample {
        input: Vec<f32>,
    }

    impl TestExample {
        fn new(input: Vec<f32>) -> Self {
            Self { input }
        }
    }

    fn max_difference(input: &[f32]) -> f32 {
        let mut max = 0.0;

        for (i, v) in input.iter().enumerate() {
            for w in input.iter().skip(i + 1) {
                let diff = (v - w).abs();
                if diff > max {
                    max = diff;
                }
            }
        }

        max
    }

    impl ClassificationExample for TestExample {
        fn get_input(&self) -> Vec<f32> {
            self.input.clone()
        }

        fn get_category(&self) -> usize {
            if max_difference(&self.input) > 0.8 {
                1
            } else {
                0
            }
        }

        fn get_categories_count(&self) -> usize {
            2
        }
    }

    fn create_simple_network() -> Network {
        let mut network = Network::new(2, ErrorFunction::EuclideanDistanceSquared);
        network
            .add_layer(
                2, true, 0.0,
                NeuronActivation::LeakyRelu(0.01),
                LayerActivation::None,
            )
            .add_layer(
                2, false, 0.0,
                NeuronActivation::Sigmoid,
                LayerActivation::SoftMax,
            );
        network
    }

    #[test]
    fn test_create_network() {
        let network = create_simple_network();
        assert_eq!(network.params.len(), 10);
    }

    #[test]
    fn test_feed_forward() {
        let mut network = create_simple_network();

        network.params = vec![0.5, 0.1, 0.3, 0.2, 0.4, 0.6, 0.15, 0.25, 0.15, 0.7];

        let input = TestExample::new(vec![0.8, 0.2]);

        let mut nf = FloatFactory::new();

        let ff = network.feed_forward(&mut nf, &input, true);

        let ia = 0.5 + 0.1 * 0.8 + 0.3 * 0.2;
        let ib = 0.2 + 0.4 * 0.8 + 0.6 * 0.2;
        let a = nf.activate_neuron(&ia, &network.layer_configs[0].neuron_activation);
        let b = nf.activate_neuron(&ib, &network.layer_configs[0].neuron_activation);
        let ic = 0.15 * a + 0.25 * b;
        let id = 0.15 * a + 0.7 * b;
        let c = nf.activate_neuron(&ic, &network.layer_configs[1].neuron_activation);
        let d = nf.activate_neuron(&id, &network.layer_configs[1].neuron_activation);
        let i_out = vec![c, d];
        let out = nf.activate_layer(&i_out, &network.layer_configs[1].layer_activation);
        let expected = input.get_expected_one_hot();
        let error = nf.compute_error(&expected, &out, &network.error_function);
        assert_eq!(ff.error, error);
    }

    #[test]
    fn test_back_propagate() {
        let cnf = || AutoDiff::new();
        let t_conf = TrainingConfig::new(5, 2, 0.01, 0.0001, 32, 4);

        let mut network = create_simple_network();
        network.params = vec![0.5, 0.1, 0.3, 0.2, 0.4, 0.6, 0.15, 0.25, 0.7, 0.2];
        let initial_params = network.params.clone();

        let samples = vec![
            TestExample::new(vec![0.1, 0.9]),
            TestExample::new(vec![0.4, 0.7]),
        ];

        let error = network.feed_batch_forward(cnf, &samples, true);

        network.back_propagate(&error.diffs, &t_conf);
        assert_ne!(initial_params, network.params);
        let error2 = network.feed_batch_forward(cnf, &samples, true);
        assert_ne!(error2.error.scalar(), error.error.scalar());
    }
}
