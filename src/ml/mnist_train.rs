use crate::ml::{
    NeuronActivation,
    LayerActivation,
    scalar_network::{
        ScalarNetwork,
        LayerConfig,
    },
    ErrorFunction,
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

}

#[test]
fn test_mnist_scalar_network() {
    let network = create_mnist_scalar_network();
    assert_eq!(network.weights(0, 0).len(), 28 * 28);
    assert_eq!(network.weights(0, 31).len(), 28 * 28);
    assert_eq!(network.weights(1, 4).len(), 32);
    assert!(network.bias(0, 16).is_some());
}
