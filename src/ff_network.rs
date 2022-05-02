use rand::prelude::*;
use rayon::prelude::*;

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

use super::autodiff::{
    AutoDiff,
    ADValue,
};

use super::math::{
    apply_neuron_activation_f32,
    apply_layer_activation_f32,
    index_of_max_value,
};

pub trait ClassificationExample {
    fn get_input(&self) -> Vec<f32>;
    fn get_label(&self) -> usize;
}

#[derive(Debug, Clone)]
struct Neuron {
    bias: Option<ADValue>,
    weights: Vec<ADValue>,
    activation: Option<NeuronActivation>,
    ad_value: ADValue,
}

#[derive(Debug, Clone)]
struct FFLayer {
    neurons: Vec<Neuron>,
    activation: Option<LayerActivation>,
}

impl FFLayer {
    fn to_vec(&self) -> Vec<ADValue> {
        self.neurons.iter().map(|n| n.ad_value).collect()
    }

    fn from_vec(&mut self, input: &Vec<ADValue>) {
        for (n, &i) in self.neurons.iter_mut().zip(input.iter()) {
            n.ad_value = i;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Network {
    autodiff: AutoDiff,
    layers: Vec<FFLayer>,
}

pub struct BatchResult {
    pub net: Network,
    pub error: ADValue,
}

impl Network {
    pub fn new() -> Network {
        Network {
            autodiff: AutoDiff::new(),
            layers: Vec::new(),
        }
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn autodiff(&mut self) -> &mut AutoDiff {
        &mut self.autodiff
    }

    pub fn add_layer(
        &mut self,
        neurons_count: usize,
        use_bias: bool,
        neuron_activation: Option<NeuronActivation>,
        layer_activation: Option<LayerActivation>,
    ) -> &mut Network {
        let mut rng = rand::thread_rng();

        let mut layer = FFLayer {
            neurons: Vec::new(),
            activation: layer_activation,
        };

        for _ in 0..neurons_count {
            let mut weights_f32: Vec<f32> = Vec::new();

            if self.layers.len() > 0 {
                let prev_layer = self.layers.last().unwrap();
                let mut total_weight = 0.0f32;
                for _ in 0..prev_layer.neurons.len() {
                    let weight = rng.gen();
                    weights_f32.push(weight);
                    total_weight += weight;
                }
                for w in weights_f32.iter_mut() {
                    *w /= total_weight;
                }
            }

            let weights = weights_f32.iter().map(|w| self.autodiff.create_variable(*w)).collect();

            let neuron = Neuron {
                weights,
                bias: if use_bias {
                    Some(self.autodiff.create_variable(rng.gen()))
                } else {
                    None
                },
                activation: neuron_activation,
                ad_value: self.autodiff.create_constant(0.0),
            };
            layer.neurons.push(neuron);
        }
        self.layers.push(layer);
        self
    }

    pub fn feed_forward(&mut self, input: &dyn ClassificationExample) -> Vec<ADValue> {
        let input_vec = input.get_input();

        if input_vec.len() != self.layers[0].neurons.len() {
            panic!("input vector length does not match the number of neurons in the first layer");
        }

        for (neuron, &input_value) in self.layers[0].neurons.iter_mut().zip(input_vec.iter()) {
            neuron.ad_value = self.autodiff.create_constant(input_value);
        }

        for l in 1..self.layers.len() {
            let (prev_layer, next_layers) = self.layers.split_at_mut(l);
            let layer = &mut next_layers[0];
            let prev_layer = &prev_layer[l - 1];

            for neuron in layer.neurons.iter_mut() {
                neuron.ad_value = if let Some(bias) = neuron.bias {
                    bias
                } else {
                    self.autodiff.create_constant(0.0)
                };

                for (&weight, connected_neuron) in neuron.weights.iter().zip(prev_layer.neurons.iter()) {
                    let contrib = self.autodiff.mul(
                        weight,
                        connected_neuron.ad_value,
                    );

                    neuron.ad_value = self.autodiff.add(
                        neuron.ad_value,
                        contrib,
                    );
                }

                if let Some(activation) = neuron.activation {
                    neuron.ad_value = self.autodiff.apply_neuron_activation(
                        neuron.ad_value,
                        &activation,
                    );
                }
            }

            if let Some(activation) = layer.activation {
                let activated = self.autodiff.apply_layer_activation(
                    &layer.to_vec(),
                    &activation,
                );

                layer.from_vec(&activated);
            }
        }

        self.layers.last().unwrap().to_vec()
    }

    pub fn compute_batch_error<T: ClassificationExample>(&mut self, examples: &Vec<&T>) -> ADValue {
        let mut batch_error = self.autodiff.create_constant(0.0);

        for example in examples.iter() {
            let output = self.feed_forward(*example);
            let mut one_hot = vec![self.autodiff.create_constant(0.0); output.len()];
            one_hot[example.get_label()] = self.autodiff.create_constant(1.0);
            let error = self.autodiff().euclidean_distance_squared(&output, &one_hot);
            batch_error = self.autodiff.add(batch_error, error);
        }

        self.autodiff.div(batch_error, self.autodiff.create_constant(examples.len() as f32))
    }

    pub fn predict(&self, input: &dyn ClassificationExample) -> usize {
        let mut activations = input.get_input();

        if activations.len() != self.layers[0].neurons.len() {
            panic!("input vector length does not match the number of neurons in the first layer");
        }

        for l in 1..self.layers.len() {
            let layer = &self.layers[l];

            let mut new_activations: Vec<_> = layer.neurons.iter().map(|n| {
                let mut activation = if let Some(bias) = n.bias {
                    bias.value
                } else {
                    0.0
                };

                activation += n.weights.iter().zip(activations.iter()).fold(0.0, |acc, (&weight, &input_value)| {
                    acc + weight.value * input_value
                });

                if let Some(fun) = n.activation {
                    activation = apply_neuron_activation_f32(activation, &fun);
                }

                activation
            }).collect();

            if let Some(fun) = layer.activation {
                new_activations = apply_layer_activation_f32(&new_activations, &fun);
            }

            activations = new_activations;
        }

        index_of_max_value(&activations)
    }

    pub fn compute_example_error(&mut self, input: &dyn ClassificationExample) -> ADValue {
        let output = self.feed_forward(input);
        let mut expected = vec![self.autodiff.create_constant(0.0); output.len()];
        expected[input.get_label()] = self.autodiff.create_constant(1.0);

        self.autodiff.euclidean_distance_squared(
            &output,
            &expected,
        )
    }

    pub fn compute_accuracy<T: ClassificationExample>(&self, examples: &Vec<T>) -> f32 {
        let mut correct = 0.0;
        for example in examples {
            if self.predict(example) == example.get_label() {
                correct += 1.0;
            }
        }

        100.0 * correct / examples.len() as f32
    }

    pub fn back_propagate(&mut self, error: ADValue, learning_rate: f32) {
        // Perform the actual back propagation
        for l in 1..self.layers.len() {
            let layer = &mut self.layers[l];
            for neuron in layer.neurons.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    let error_contrib = self.autodiff.diff(
                        error,
                        *weight,
                    );
                    weight.value -= learning_rate * error_contrib;
                }

                if let Some(mut bias) = neuron.bias {
                    let error_contrib = self.autodiff.diff(
                        error,
                        bias,
                    );
                    bias.value -= learning_rate * error_contrib;
                }
            }
        }

        self.reset();
    }

    pub fn average(&mut self, nets: &Vec<Network>) {
        for l in 0..self.layers.len() {
            let layer = &mut self.layers[l];
            for (n, neuron) in layer.neurons.iter_mut().enumerate() {
                for (w, weight) in neuron.weights.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for net in nets.iter() {
                        sum += net.layers[l].neurons[n].weights[w].value;
                    }
                    weight.value = sum / nets.len() as f32;
                }

                if let Some(mut bias) = neuron.bias {
                    let mut sum = 0.0;
                    for net in nets.iter() {
                        sum += net.layers[l].neurons[n].bias.unwrap().value;
                    }
                    bias.value = sum / nets.len() as f32;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        // Avoid explosion of the number of tracked variables
        self.autodiff.reset();

        // Prepare the network again for next automatic differentiation
        for l in 1..self.layers.len() {
            let layer = &mut self.layers[l];
            for neuron in layer.neurons.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    *weight = self.autodiff.create_variable(weight.value);
                }

                if let Some(bias) = neuron.bias {
                    neuron.bias = Some(self.autodiff.create_variable(bias.value));
                }
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_forward_prop() {
        struct XORExample {
            x: f32,
            y: f32,
        }

        impl XORExample {
            fn new() -> Self {
                let mut rng = rand::thread_rng();

                XORExample {
                    x: rng.gen(),
                    y: rng.gen(),
                }
            }
        }

        impl ClassificationExample for XORExample {
            fn get_input(&self) -> Vec<f32> {
                vec![self.x, self.y]
            }

            fn get_label(&self) -> usize {
                let (x, y): (bool, bool) = (self.x > 0.9, self.y > 0.9);
                if x && y {
                    0
                } else if x || y {
                    1
                } else {
                    0
                }
            }
        }

        let mut network = Network::new();
        network
            .add_layer(2, false, None, None)
            .add_layer(2, false, Some(NeuronActivation::LeakyReLU(0.01)), None)
            .add_layer(2, false, Some(NeuronActivation::LeakyReLU(0.01)), None)
        ;

        let a = network.feed_forward(&XORExample::new());
        let b = network.feed_forward(&XORExample::new());

        assert_ne!(a[0].value, b[0].value);
        assert_ne!(a[1].value, b[1].value);
    }

}
