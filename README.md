# ml-rust

A project where I experiment with machine learning and Rust, learning everything from scratch and not using any third-party library for machine learning.

Currently I have a reasonably efficient MNIST recognizer working, with around 95% accuracy on the testing set, using a 2 layers feed-forward neural network with 30 (biased) hidden neurons activated by reLu, softMax on the output layer and categorical cross entropy as a loss function, so I'm rather happy about the progress so far.

For MNIST, training is done in batches, whose size reduces slightly after each batch, as well as the learning rate.

Optimization is made via stochastic gradient descent, using (self implemented) automatic differentiation for computing error gradients and parameter updates.
