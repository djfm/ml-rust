use std::{
    time::{
        Instant,
    },
};

use rayon::prelude::*;
use rand::prelude::*;

use crate::ml::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
    ClassificationExample,
    NumberFactory,
    DifferentiableNumberFactory,
    NumberLike,
    FloatFactory,
    ADFactory,
    util::{
        average_scalar_vectors,
        human_duration,
        windows,
        WindowIteratorConfig,
    },
    trainer::{
        TrainingConfig,
    },
};

pub struct LayerConfig {
    pub use_biases: bool,
    pub neurons_count: usize,
    pub neuron_activation: NeuronActivation,
    pub layer_activation: LayerActivation,
}

impl LayerConfig {
    pub fn new(neurons_count: usize) -> Self {
        Self {
            use_biases: true,
            neurons_count,
            neuron_activation: NeuronActivation::LeakyReLU(0.01),
            layer_activation: LayerActivation::None,
        }
    }
}

pub struct Network {
    input_size: usize,
    params: Vec<f32>,
    layers: Vec<LayerConfig>,
    error_function: ErrorFunction,
    params_offsets: Vec<usize>,
}

pub struct TrainingResult<N> where N: NumberLike {
    pub error: N,
    pub params: Vec<N>,
    pub diffs: Vec<f32>,
    pub label: Option<usize>,
}

impl <N: NumberLike> TrainingResult<N> {
    pub fn new<F: NumberFactory<N>>(nf: &mut F) -> Self {
        TrainingResult {
            error: nf.create_variable(0.0),
            params: vec![],
            diffs: vec![],
            label: None,
        }
    }
}

impl Network {
    pub fn new(input_size: usize, error_function: ErrorFunction, layers: Vec<LayerConfig>) -> Self {
        let mut rng = thread_rng();
        let mut params = Vec::new();
        let mut prev_size = input_size;
        let mut params_offsets = Vec::new();
        params_offsets.push(0);

        for layer in &layers {
            let size = (prev_size + if layer.use_biases { 1 } else { 0 }) * layer.neurons_count ;

            for _ in 0..size {
                params.push(rng.gen());
            }

            params_offsets.push(size);

            prev_size = layer.neurons_count;
        }

        Network {
            input_size,
            params,
            layers,
            error_function,
            params_offsets,
        }
    }

    pub fn weights(&self, layer: usize, neuron: usize) -> &[f32] {
        let prev_size = if layer == 0 {
            self.input_size
        } else {
            self.layers[layer - 1].neurons_count
        };

        let use_biases = if self.layers[layer].use_biases { 1 } else { 0 };
        let layer_offset = if layer == 0 { 0 } else { self.params_offsets[layer - 1] };
        let offset = layer_offset + neuron * (prev_size + use_biases);

        &self.params[offset..offset + prev_size]
    }

    pub fn bias(&self, layer: usize, neuron: usize) -> Option<f32> {
        let offset = self.params_offsets[layer] + neuron * self.layers[layer].neurons_count;

        if self.layers[layer].use_biases {
            Some(self.params[offset])
        } else {
            None
        }
    }

    pub fn pass_forward<S, F, N>(
        &self,
        sample: &S,
        nf: &mut F,
    ) -> TrainingResult<N>
    where
        S: ClassificationExample,
        F: NumberFactory<N>,
        N: NumberLike,
    {
        let mut res = TrainingResult::new(nf);
        let mut params = vec![nf.create_variable(0.0); self.params.len()];

        let mut input = sample.get_input().iter().map(|x| nf.create_variable(*x)).collect::<Vec<_>>();
        let expected = sample.get_one_hot().iter().map(|x| nf.create_variable(*x)).collect::<Vec<_>>();

        for l in 0..self.layers.len() {
            let conf = &self.layers[l];

            let next_layer = (0..conf.neurons_count).map(|n| {
                let mut neuron_input = if let Some(bias) = self.bias(l, n) {
                    let bias = nf.create_variable(bias);

                    if nf.has_automatic_diff() {
                        params.push(bias);
                    }

                    bias
                } else {
                    nf.create_variable(0.0)
                };

                self.weights(l, n).iter().zip(input.iter()).for_each(|(w, x)| {
                    let w = nf.create_variable(*w);

                    if nf.has_automatic_diff() {
                        params.push(w);
                    }

                    let p = nf.multiply(w, *x);
                    neuron_input = nf.addition(neuron_input, p);
                });

                nf.activate_neuron(neuron_input, &conf.neuron_activation)
            }).collect::<Vec<_>>();

            if conf.layer_activation != LayerActivation::None {
                input = nf.activate_layer(&next_layer, &conf.layer_activation)
            } else {
                input = next_layer;
            }
        }

        res.params = params;
        res.error = nf.compute_error(&expected, &input, &self.error_function);
        res.label = Some(nf.get_label(&input));

        res
    }

    pub fn get_batch_diffs<S>(
        &self,
        samples: &[S],
    ) -> Vec<f32>
    where
        S: ClassificationExample,
    {
        let forward_results = samples.par_iter().map(|s| {
            let mut nf = ADFactory::new();
            let fwd = self.pass_forward(s, &mut nf);
            self.get_params_diffs(&fwd, &mut nf)
        }).collect::<Vec<Vec<f32>>>();

        average_scalar_vectors(&forward_results)
    }

    pub fn get_params_diffs<F, N>(
        &self,
        error: &TrainingResult<N>,
        nf: &mut F,
    ) -> Vec<f32>
    where
        F: DifferentiableNumberFactory<N>,
        N: NumberLike,
    {
        error.params.iter().map(|p| nf.diff(error.error, *p)).collect::<Vec<_>>()
    }

    pub fn predict<S: ClassificationExample>(&self, input: &S) -> usize {
        let mut ff = FloatFactory::new();
        let res = self.pass_forward(input, &mut ff);
        res.label.expect("prediction has a label")
    }

    pub fn compute_batch_accuracy<T: ClassificationExample>(&self, examples: &[T]) -> f32 {
        let mut correct = 0;
        let total = examples.len();

        for example in examples {
            let prediction = self.predict(example);
            if prediction == example.get_label() {
                correct += 1;
            }
        }

        100.0 * correct as f32 / total as f32
    }

    fn update_params(&mut self, diffs: &[f32], tconf: &TrainingConfig) {
        for (i, p) in self.params.iter_mut().enumerate() {
            *p -= diffs[i] * tconf.learning_rate;
        }
    }

    pub fn ad_train<S: ClassificationExample>(
        &mut self,
        training: &[S],
        testing: &[S],
        config: &TrainingConfig,
    ) -> &mut Self {
        let start = Instant::now();
        let mut processed = 0;
        let mut tconf = config.clone();
        let total = training.len() * tconf.epochs;

        let window_config = WindowIteratorConfig::new(tconf.batch_size());

        for epoch in 1..=tconf.epochs {
            for batch in windows(training, &window_config) {
                let diffs = self.get_batch_diffs(&batch);

                self.update_params(&diffs, &tconf);

                tconf.update(batch.len());
                window_config.set_size(tconf.batch_size());

                processed += batch.len();
                let accuracy = self.compute_batch_accuracy(batch);
                println!(
                    "\nEpoch {}/{}: {}={:.2}% processed.",
                    epoch, tconf.epochs,
                    processed, 100.0 * processed as f32 / total as f32,
                );

                println!(
                    "batch size is {}, batch accuracy: {:.2}%, learning rate: {:.6}",
                    batch.len(), accuracy,
                    tconf.learning_rate,
                );
            }

            let testing_accuracy = self.compute_batch_accuracy(&testing);
            println!("\nTesting accuracy after epoch {}: {:.2}%\n", epoch, testing_accuracy);
        }

        println!("\nTraining finished in {:?}", human_duration(start.elapsed()));

        self
    }
}

#[test]
fn test_weights_biases_layout() {
    let net = Network::new(
        2, ErrorFunction::EuclideanDistanceSquared,
        vec![
            LayerConfig::new(2),
            LayerConfig {
                use_biases: false,
                layer_activation: LayerActivation::SoftMax,
                ..LayerConfig::new(1)
            },
        ],
    );

    assert_eq!(net.weights(0, 0).len(), 2);
    assert_eq!(net.weights(0, 1).len(), 2);
    assert_eq!(net.weights(1, 0).len(), 2);

    assert_eq!(net.bias(0, 0).is_some(), true);
    assert_eq!(net.bias(0, 1).is_some(), true);
    assert_eq!(net.bias(1, 0).is_none(), true);

    assert_eq!(net.params.len(), 9);
}
