# ml-rust

A project where I experiment with machine learning and Rust, learning everything from scratch and not using any third-party library for machine learning.

Currently I have a reasonably effificent MNIST recognizer working, with > 93% accuracy, using a 2 layers perceptron with 30 (biased) hidden neurons activated by reLu, softMax on the output layer and categorical cross entropy as a loss function, so I'm rather happy about the progres

For MNIST training is done in batches, whose size reduces slightly after each batch, as well as the learning rate.
