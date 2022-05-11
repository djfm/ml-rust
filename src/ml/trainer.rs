use crate::ml::{
    DifferentiableNumberFactory,
    NumberFactory,
    NumberLike,
    ADFactory, ADNumber,
    ClassificationExample,
    scalar_network::{
        ScalarNetwork,
    }
};

pub struct TrainingConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
}

impl TrainingConfig {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 5,
            batch_size: 32,
        }
    }
}

pub fn compute_params_diffs(network: &ScalarNetwork, sample: &dyn ClassificationExample) -> Vec<f32> {
    let mut ad = ADFactory::new();
    let mut params = Vec::new();

    let mut previous_activations = sample.get_input().iter().map(
        |x| ad.create_variable(*x)
    ).collect::<Vec<_>>();

    let expected = sample.get_one_hot().iter().map(
        |x| ad.create_variable(*x)
    ).collect::<Vec<_>>();

    for l in 0..network.layers_count() {
        let config = network.layer_config(l);

        let activations = (0..config.neurons_count).into_iter().map(|n| {
            let mut sum = if let Some(bias) = network.bias(l, n) {
                let b = ad.create_variable(bias);
                params.push(b);
                b
            } else {
                ad.create_variable(0.0)
            };

            for (w, a) in network.weights(l, n).iter().zip(previous_activations.iter()) {
                let w = ad.create_variable(*w);
                params.push(w);
                let m = ad.multiply(w, *a);
                sum = ad.addition(sum, m);
            }

            ad.activate_neuron(sum, &config.neuron_activation)
        }).collect::<Vec<ADNumber>>();

        previous_activations = ad.activate_layer(&activations, &config.layer_activation);
    }

    let error = ad.compute_error(&expected, &previous_activations, &network.error_function());

    params.iter().map(|x| ad.diff(error, *x)).collect()
}

pub fn compute_batch_diffs<S: ClassificationExample>(
    network: &ScalarNetwork,
    samples: &[S],
) -> Vec<f32> {
    let diffs = samples.iter().map(|s| compute_params_diffs(network, s)).collect::<Vec<_>>();
    let mut res = vec![0.0; network.params_count()];

    for d in diffs.iter() {
        if d.len() != res.len() {
            println!("{:?}", res.len());
            panic!("Invalid diff length");
        }

        for (i, v) in d.iter().enumerate() {
            res[i] += v / samples.len() as f32;
        }
    }

    res
}

pub fn update_network(
    network: &mut ScalarNetwork,
    training_config: &TrainingConfig,
    diffs: &[f32],
) {
    for (p, d) in network.params().iter_mut().zip(diffs.iter()) {
        *p -= training_config.learning_rate * *d;
    }
}
