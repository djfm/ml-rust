use lib::ml;
use lib::ml::data;

use lib::ml::{
    Network,
    ErrorFunction,
    NeuronActivation,
    LayerActivation,
    TrainingConfig,
};

pub fn create_network() -> Network {
    let mut network = Network::new(28 * 28, ErrorFunction::CategoricalCrossEntropy);

    network
        .add_layer(32, true, NeuronActivation::LeakyRelu(0.1), LayerActivation::None)
        .add_layer(10, false, NeuronActivation::None, LayerActivation::SoftMax)
    ;

    network
}

pub fn train() -> Network {
    match (data::mnist::load_training_set("lib"), data::mnist::load_testing_set("lib")) {
        (Ok(mut training_set), Ok(testing_set)) => {
            let mut network = create_network();
            let t_conf = TrainingConfig::new(5, training_set.len(), 0.01, 0.0001, 128, 8);
            ml::train(&mut network, &mut training_set, &testing_set, t_conf);
            network
        },
        (Err(e), _) => panic!("Failed to load the training set: {}", e),
        (_, Err(e)) => panic!("Failed to load the testing set: {}", e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib::ml::{
        data::mnist,
        AutoDiff,
        FloatFactory,
    };

    #[test]
    fn test_compute_the_same_with_different_number_factories() {
        let ts = mnist::load_training_set("lib").expect("the mnist training set should be available");
        let input = &ts[0];
        let net = create_network();

        let mut ad = AutoDiff::new();
        let ad_error = net.feed_forward(&mut ad, input);

        let mut ff = FloatFactory::new();
        let ff_error = net.feed_forward(&mut ff, input);

        assert_eq!(ad_error.error(), ff_error.error());
    }
}
