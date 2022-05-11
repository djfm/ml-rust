use crate::ml::{
    NeuronActivation,
    LayerActivation,
    scalar_network::{
        ScalarNetwork,
        LayerConfig,
    },
    ErrorFunction,
    trainer::{
        TrainingConfig,
        compute_batch_diffs,
        update_network,
    },
    mnist::{
        load_training_set,
        load_testing_set,
    },
};

pub fn create_mnist_scalar_network() -> ScalarNetwork {
    ScalarNetwork::new(
        28 * 28, ErrorFunction::EuclideanDistanceSquared,
        vec![
            LayerConfig {
                neurons_count: 32,
                layer_activation: LayerActivation::None,
                neuron_activation: NeuronActivation::LeakyReLU(0.01),
                use_biases: true,
            },
            LayerConfig {
                neurons_count: 10,
                layer_activation: LayerActivation::SoftMax,
                neuron_activation: NeuronActivation::LeakyReLU(0.01),
                use_biases: true,
            }
        ],
    )
}

pub fn train() {
    let tconf = TrainingConfig::new();
    let mut network = create_mnist_scalar_network();

    let training_set = match load_training_set() {
        Ok(set) => set,
        Err(err) => {
            println!("Could not load training set: {}", err);
            std::process::exit(1);
        }
    };

    let testing_set = match load_testing_set() {
        Ok(set) => set,
        Err(err) => {
            println!("Could not load testing set: {}", err);
            std::process::exit(1);
        }
    };

    let mut processed = 0;
    let total = training_set.len() * tconf.epochs;

    for epoch in 1..=tconf.epochs {
        for batch in training_set.windows(tconf.batch_size) {
            let diffs = compute_batch_diffs(&network, batch);
            update_network(&mut network, &tconf, &diffs);
            processed += batch.len();
            let error = network.compute_batch_error(batch);
            println!(
                "epoch {}: {:.2}% processed. Batch error: {:.2}",
                epoch,
                100.0 * processed as f32 / total as f32,
                error,
            );
        }
    }
}

#[test]
fn test_mnist_scalar_network() {
    let network = create_mnist_scalar_network();
    assert_eq!(network.weights(0, 0).len(), 28 * 28);
    assert_eq!(network.weights(0, 31).len(), 28 * 28);
    assert_eq!(network.weights(1, 4).len(), 32);
    assert!(network.bias(0, 16).is_some());
}
