use ml_rust::data::mnist_loader;

use ml_rust::{
    Network,
    ErrorFunction,
    NeuronActivation,
    LayerActivation,
    TrainingConfig,
};

pub fn create_network() -> Network {
    let mut network = Network::new(28 * 28, ErrorFunction::CategoricalCrossEntropy);

    network
        .add_layer(
            32, true, 0.5,
            NeuronActivation::LeakyRelu(0.01), LayerActivation::None
        )
        .add_layer(
            10, false, 0.0,
            NeuronActivation::None, LayerActivation::SoftMax
        )
    ;

    network
}

pub fn train() -> Network {
    match (mnist_loader::load_training_set("data"), mnist_loader::load_testing_set("data")) {
        (Ok(mut training_set), Ok(testing_set)) => {
            let mut network = create_network();
            let t_conf = TrainingConfig::new(
                10, training_set.len(),
                0.01, 0.0001,
                128, 8,
            );
            ml_rust::train(&mut network, &mut training_set, &testing_set, t_conf);
            network
        },
        (Err(e), _) => panic!("Failed to load the training set: {}", e),
        (_, Err(e)) => panic!("Failed to load the testing set: {}", e)
    }
}

pub fn main() {
    train();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_rust::{
        AutoDiff,
        FloatFactory,
    };

    #[test]
    fn test_compute_the_same_with_different_number_factories() {
        let ts = mnist_loader::load_training_set("data").expect("the mnist training set should be available");
        let input = &ts[0];
        let net = create_network();

        let mut ad = AutoDiff::new();
        let ad_error = net.feed_forward(&mut ad, input, true);

        let mut ff = FloatFactory::new();
        let ff_error = net.feed_forward(&mut ff, input, true);

        assert_eq!(ad_error.error(), ff_error.error());
    }
}
