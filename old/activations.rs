#[derive(Copy, Clone, Debug)]
pub enum NeuronActivation {
    LeakyReLU(f32),
}

#[derive(Copy, Clone, Debug)]
pub enum LayerActivation {
    SoftMax,
}
