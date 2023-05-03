# ml-rust

A project where I experiment with machine learning and Rust, learning everything from scratch and not using any third-party library for machine learning.

Currently I have a reasonably efficient MNIST recognizer working, with around 96% accuracy on the testing set, using a 2 layers feed-forward neural network with 30 (biased) hidden neurons activated by reLu, softMax on the output layer and categorical cross entropy as a loss function, so I'm rather happy about the progress so far.

For MNIST, training is done in batches, whose size reduces slightly after each batch, as well as the learning rate.

Optimization is made via stochastic gradient descent, using (self implemented) automatic differentiation for computing error gradients and parameter updates.

# Here is an example of the MNIST recognizer running on a decent laptop:
![ml-rust-mnist](https://user-images.githubusercontent.com/1460499/175834092-24ff2c17-474f-4162-a2d5-8be0ecd67489.png)

The jagged lines represent the accuracy on each mini batch, while the more infrequent segments represent the acccuracy on the testing set after each training epoch is completed.

# Updates, things I learned and pitfalls I've encountered along the way

## Dropout

I've added a 50% dropout on the hidden layer (during training only obviously) and I'm quite pleased with the results:

- testing accuracy is slightly lower with the same number of epochs,
  but since training an epoch is much faster now I can run many more epochs during training and still remain under 20 minutes
  to complete training on my old 8-pseudocores desktop

- on that same old hardware with dropout I get 96% accuracy on the testing set on average and often above that (typically I use 10 epochs)

## Problems Identified

### NaN's

They're pain in the a** :)

Most of them seem to result from computing large softmaxes, even though I'm using the so called [softmax trick](https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/)

My strategy currently when I can't figure out where the NaNs are coming from is to replace them with either a very big or very small numbers if they're infinity, and, if not, pick a value at random, but that doesn't seem to work very well.

I'm wondering whether simply disconnecting a NaN neuron from the network would work, after all, it's dead.

# Trying it out: how to install and run

**Warning**: Only tested on linux.

## Prerequisites

On debian based distributions, which is all that I have tested,
you will probably need `build-essential`,
and the only "exotic" dependency is libSDL2, that you should be able
to install pretty universally with:

```bash
sudo apt install libsdl2-dev
```

## Compile and run

```bash
git clone git@github.com:djfm/ml-rust.git
cd ml-rust
cargo run --release
```
