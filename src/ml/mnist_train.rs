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
        train_scalar_network,
    },
    mnist::{
        load_training_set,
        load_testing_set,
    },
};

pub fn create_network() -> ScalarNetwork {
    ScalarNetwork::new(
        28 * 28,
        ErrorFunction::EuclideanDistanceSquared,
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
                neuron_activation: NeuronActivation::None,
                use_biases: false,
            }
        ],
    )
}

pub fn train() -> ScalarNetwork {
    let tconf = TrainingConfig::new();
    let mut network = create_network();

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

    train_scalar_network(&mut network, &tconf, &training_set, &testing_set);

    network
}

#[test]
fn test_mnist_scalar_network() {
    let network = create_network();
    assert_eq!(network.weights(0, 0).len(), 28 * 28);
    assert_eq!(network.weights(0, 31).len(), 28 * 28);
    assert_eq!(network.weights(1, 4).len(), 32);
    assert!(network.bias(0, 16).is_some());
}
