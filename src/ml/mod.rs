mod number_factory;
mod number_like;
mod network;
mod layer;
mod math;
mod mnist;
mod ad_factory;
mod ad_number;

pub use number_factory::{
    NumberFactory,
};

pub use number_like::{
    NumberLike,
};

pub use network::{
    Network,
};

pub use layer::{
    Layer,
    LayerConfig,
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

pub trait ClassificationExample {
    fn get_input(&self) -> Vec<f32>;
    fn get_label(&self) -> usize;
}
