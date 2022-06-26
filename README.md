# ml-rust

A project where I experiment with machine learning and Rust, learning everything from scratch and not using any third-party library for machine learning.

Currently I have a reasonably efficient MNIST recognizer working, with around 95% accuracy on the testing set, using a 2 layers feed-forward neural network with 30 (biased) hidden neurons activated by reLu, softMax on the output layer and categorical cross entropy as a loss function, so I'm rather happy about the progress so far.

For MNIST, training is done in batches, whose size reduces slightly after each batch, as well as the learning rate.

Optimization is made via stochastic gradient descent, using (self implemented) automatic differentiation for computing error gradients and parameter updates.

## Upodates and pitfalls as I learn

### Dropout

I've added a 50% dropout (during training only obviously) and I'm quite please with the results:

- testing accuracy is slightly lower- but being faster I can run many more epochs during training and sill remain under 20 minutes on my old 8-
  pseudocores desktop

- on that same old hardware with dropout I gat 96% accujracy on the testinhg set on average an often abovt thsat.

## Problems Identified

### NaN's

They're pain in the a** :)

Most of themresult from computing lage softtmaxes, even thoihj I'm using the so called [softmax trick](https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/)

My strategy currentlu when I can't figure out where the nans are coming from is to replace them with either very big or very small numbers if they're infinity; and if not, pick a value at random but that doesn' seem to work very well.

# Testing

**Warning**: Only tested on linux.

```bash
git clone git@github.com:djfm/ml-rust.git
cd ml-rust
cargo run --release
```
