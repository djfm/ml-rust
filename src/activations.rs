#[derive(Copy, Clone)]
pub enum NeuronActivation {
    LeakyReLU(f32),
}

#[derive(Copy, Clone)]
pub enum LayerActivation {
    SoftMax,
}
