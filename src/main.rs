mod ml;

pub fn main() {
    let mut net = ml::mnist_train::create_network();
    ml::mnist_train::train(&mut net);
}
