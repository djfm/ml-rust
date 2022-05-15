use rayon::prelude::*;

mod number_factory;
mod number_like;
mod math;
mod mnist;
mod ad_factory;
mod ad_number;
mod float_factory;
mod network;
mod trainer;
pub mod mnist_train;
pub mod util;

pub use number_factory::{
    NumberFactory,
    DifferentiableNumberFactory,
};

pub use number_like::{
    NumberLike,
};

pub use math::{
    NeuronActivation,
    LayerActivation,
    ErrorFunction,
};

pub use mnist::{
    load_training_set,
    load_testing_set,
};

pub use ad_factory::{
    ADFactory,
};

pub use ad_number::{
    ADNumber,
};

pub use float_factory::{
    FloatFactory,
};

pub use network::{
    Network,
    LayerConfig,
};

pub use trainer::{
    TrainingConfig,
};

pub trait ClassificationExample: Sync {
    fn get_input(&self) -> Vec<f32>;
    fn get_label(&self) -> usize;
    fn get_one_hot(&self) -> Vec<f32> {
        let mut expected = vec![0.0; self.get_input().len()];
        expected[self.get_label()] = 1.0;
        expected
    }
}
