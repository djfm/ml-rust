use crate::ml::{
    Network,
    ErrorFunction,
    NeuronActivation,
    LayerActivation,
};

pub fn create_network() -> Network {
    let mut network = Network::new(28 * 28, ErrorFunction::CategoricalCrossEntropy);

    network
        .add_layer(30, true, NeuronActivation::LeakyRelu(0.1), LayerActivation::None)
        .add_layer(10, false, NeuronActivation::Sigmoid, LayerActivation::SoftMax)
    ;

    network
}
