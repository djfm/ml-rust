mod autodiff;
mod number_factory;
mod number_like;
mod network;

pub use number_factory::{
    NumberFactory,
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
