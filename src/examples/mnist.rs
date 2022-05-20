use crate::ml::{
    Network,
    ErrorFunction,
    NeuronActivation,
    LayerActivation,
};

pub fn create_network() -> Network {
    let mut network = Network::new(28 * 28, ErrorFunction::CategoricalCrossEntropy);

    network
        .add_layer(32, true, NeuronActivation::LeakyRelu(0.1), LayerActivation::None)
        .add_layer(10, false, NeuronActivation::None, LayerActivation::SoftMax)
    ;

    network
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::data::mnist;
    use crate::ml::{
        AutoDiff,
        FloatFactory,
    };

    #[test]
    fn test_compute_the_same_with_different_number_factories() {
        let ts = mnist::load_training_set().expect("the mnist training set should be available");
        let input = &ts[0];

        let mut ad = AutoDiff::new();
        let ad_net = create_network();
        let ad_error = ad_net.feed_forward(&mut ad, input);

        let mut ff = FloatFactory::new();
        let ff_net = create_network();
        let ff_error = ff_net.feed_forward(&mut ff, input);

        assert_eq!(ad_error.error(), ff_error.error());
    }
}
