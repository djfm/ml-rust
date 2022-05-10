mod mnist;
mod ff_network;
mod activations;
mod autodiff;
mod math;
mod util;

fn main() {
    mnist::train();
    // mnist::train_parallel();
}
