use crate::{
    mnist::{
        load_training_set,
        load_testing_set,
        Image,
    },
    ml::{
        ff_network::{
            Network,
            LayerConfig,
            TrainingSample,
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
            NumberFactory,
        }
    },
};

impl <'a> TrainingSample<ADNumber<'a>, ADNumberFactory> for Image {
    fn get_input(&self) -> Vec<ADNumber<'a>> {
        self.pixels.iter().map(
            |pixel| ADNumberFactory::from_scalar(*pixel as f32 / 255.0)
        ).collect()
    }

    fn get_label(&self) -> usize {
        self.label as usize
    }

    fn get_expected_one_hot(&self) -> Vec<ADNumber<'a>> {
        let mut expected = vec![ADNumberFactory::zero(); 10];
        expected[self.label as usize] = ADNumberFactory::one();
        expected
    }
}

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

pub fn train() {
    let mut network = create_network();

    let training_set = match load_testing_set() {
        Ok(set) => set,
        Err(err) => {
            println!("Could not load training set: {}", err);
            std::process::exit(1);
        }
    };

    network.feed_forward(&training_set[0]);
}
