#[derive(Copy, Clone)]
pub enum NeuronActivation {
    None,
    LeakyReLU(f32),
}

#[derive(Copy, Clone)]
pub enum LayerActivation {
    None,
    SoftMax,
}
