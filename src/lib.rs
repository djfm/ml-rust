pub mod data;
pub mod plotter;
pub mod util;
pub mod network;
pub mod number_factory;
pub mod float_factory;
pub mod autodiff;
pub mod training;

pub use network::{
    Network,
    BatchResult,
    ClassificationExample,
};

pub use number_factory::{
    NumberFactory,
    ErrorFunction,
    LayerActivation,
    NeuronActivation,
    NumberLike,
    DifferentiableNumberFactory,
};

pub use training::{
    train,
    TrainingConfig,
};

pub use autodiff::{
    AutoDiff,
};

pub use float_factory::{
    FloatFactory,
};
