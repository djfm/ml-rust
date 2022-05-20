mod ml;
mod util;
mod examples;

use ml::{
    data::mnist,
    TrainingConfig,
};

pub fn main() {
    match (mnist::load_training_set(), mnist::load_testing_set()) {
        (Ok(training_set), Ok(testing_set)) => {
            let mut network = examples::mnist::create_network();
            let tconf = TrainingConfig { ..Default::default() };
            ml::train(&mut network, &training_set, &testing_set, &tconf);
        },
        (Err(e), _) => println!("Failed to load the training set: {}", e),
        (_, Err(e)) => println!("Failed to load the testing set: {}", e)
    }

}
