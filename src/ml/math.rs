#[derive(Copy, Clone, Debug)]
pub enum NeuronActivation {
    None,
    Relu,
    LeakyReLU(f32),
}

#[derive(Copy, Clone, Debug)]
pub enum LayerActivation {
    None,
    SoftMax,
}

#[derive(Copy, Clone, Debug)]
pub enum ErrorFunction {
    EuclideanDistanceSquared,
}
