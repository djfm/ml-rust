#[derive(Copy, Clone, Debug, PartialEq)]
pub enum NeuronActivation {
    None,
    ReLu,
    LeakyReLU(f32),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayerActivation {
    None,
    SoftMax,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ErrorFunction {
    EuclideanDistanceSquared,
}
