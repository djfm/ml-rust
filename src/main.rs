mod autodiff;
mod ff_network;
mod mnist;
mod activations;
mod graphics;
mod ml;
mod math;
mod mnist_train_2;

fn main() {
    // mnist::train();
    mnist::train_parallel();
}
