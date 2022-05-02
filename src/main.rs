mod autodiff;
mod ff_network;
mod mnist;
mod activations;
mod graphics;
mod math;
mod ad;

fn main() {
    // mnist::train();
    mnist::train_parallel();
}
