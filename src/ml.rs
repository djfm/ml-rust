mod autodiff;
mod number_factory;
mod number_like;
mod float_factory;
mod network;

pub use number_factory::{
    NumberFactory,
    DifferentiableNumberFactory,
    NumberFactoryWrapper,
    LayerActivation,
    NeuronActivation,
    ErrorFunction,
};

pub use number_like::{
    NumberLike,
};

pub use network::{
    Network,
};

pub use autodiff::{
    AutoDiff,
    ADNumber,
    DiffDefinerHelper,
};

pub use float_factory::{
    FloatFactory,
};
