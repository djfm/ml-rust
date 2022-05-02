mod autodiff;
mod ff_network;
mod mnist;
mod activations;
mod graphics;
mod math;

fn main() {
    // mnist::train();
    mnist::train_parallel();
}
