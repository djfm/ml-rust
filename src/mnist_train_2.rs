use crate::{
    mnist::{
        load_training_set,
        load_testing_set,
    },
    ml::{
        ff_network::{
            Network,
            LayerConfig,
        },
        ad::{
            AD,
        },
        ad_number::{
            ADNumber,
            ADNumberFactory,
        },
        math::{
            CellActivation,
            LayerActivation,
            ErrorFunction,
        }
    },
};

pub fn create_network<'a>() -> Network<ADNumber<'a>, ADNumberFactory> {
    let mut network = Network::new();

    network
        .add_layer(
            LayerConfig {
                ..LayerConfig::new(28 * 28)
            }
        )
        .add_layer(
            LayerConfig {
                cell_activation: CellActivation::LeakyReLU(0.01),
                ..LayerConfig::new(28 * 28)
            }
        )
        .add_layer(
            LayerConfig {
                cell_activation: CellActivation::LeakyReLU(0.01),
                layer_activation: LayerActivation::SoftMax,
                ..LayerConfig::new(28 * 28)
            }
        )
        .set_error_function(
            ErrorFunction::EuclideanDistanceSquared
        )
    ;

    network
}
