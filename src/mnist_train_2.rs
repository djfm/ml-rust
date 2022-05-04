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
            TrainingConfig,
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
        },
    },
    util::{
        human_duration,
    }
};

use std::time::{
    Instant
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
                input_size: 28 * 28,
                cell_activation: CellActivation::LeakyReLU(0.01),
                ..LayerConfig::new(32)
            }
        )
        .add_layer(
            LayerConfig {
                cell_activation: CellActivation::LeakyReLU(0.01),
                layer_activation: LayerActivation::SoftMax,
                ..LayerConfig::new(10)
            }
        )
        .set_error_function(
            ErrorFunction::EuclideanDistanceSquared
        )
    ;

    network
}

pub fn train() {
    let start_instant = Instant::now();
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

    let tconf = TrainingConfig::new();
    let mut processed = 0;
    let total = training_set.len() * tconf.epochs;

    network.feed_forward(&training_set[0]);


    // TODO: randomize between epochs
    for epoch in 1..=tconf.epochs {
        println!("Epoch {}", epoch);
        for samples in training_set.windows(tconf.batch_size) {
            let error = network.compute_batch_error(samples);
            println!("batch of size {}, error: {}", samples.len(), error.scalar());
            network.back_propagate(&error, &tconf);
            processed += samples.len();
            println!(
                "in epoch {}, {:.2}% of total processed...",
                epoch,
                processed as f32 * 100.0 / total as f32
            );
            let accuracy = network.compute_accuracy(&testing_set);
            println!("Accuracy: {}", accuracy);
        }
    }

    println!("Training complete! (in {})", human_duration(start_instant.elapsed()));
}
