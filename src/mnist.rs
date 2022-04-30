use std::fs::OpenOptions;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use super::ff_network::{
    Network,
    Example,
};


use super::autodiff::{
    AutoDiff,
};

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

pub struct Image {
    pub pixels: Vec<u8>,
    pub label: u8,
}

impl Example for Image {
    fn get_input(&self) -> Vec<f32> {
        self.pixels.iter().map(|&x| x as f32 / 255.0).collect()
    }

    fn get_one_hot_label(&self) -> Vec<f32> {
        let mut label = vec![0.0; 10];
        label[self.label as usize] = 1.0;
        label
    }
}

macro_rules! read_bytes {
    ($reader:expr, $n_bytes:expr, $what:expr) => {
        {
            let mut buf = [0u8; $n_bytes];
            match $reader.read_exact(&mut buf) {
                Ok(_) => {
                    buf.iter().fold(0, |acc, &x| acc * 256 + x as usize)
                },
                Err(e) => {
                    return Err(format!("Could not read {}: {}", $what, e));
                }
            }
        }
    };
}

pub fn load_labels(path: &str) -> Result<Vec<u8>, String> {
    match OpenOptions::new().read(true).open(path) {
        Err(e) => Err(format!("Could not open file {}: {}", path, e)),
        Ok(file) => {
            let mut reader = BufReader::new(file);

            match reader.seek(std::io::SeekFrom::Start(2)) {
                Err(e) => Err(format!("Could not seek in file {}: {}", path, e)),
                Ok(_) => {
                    let encoding = read_bytes!(reader, 1, "the data encoding byte");
                    if encoding != 0x08 {
                        return Err(format!("Unexpected encoding {}", encoding));
                    }

                    let dimensions = read_bytes!(reader, 1, "the data dimensions byte");
                    if dimensions != 0x01 {
                        return Err(format!("Unexpected dimensions {}", dimensions));
                    }

                    let n_labels = read_bytes!(reader, 4, "the number of labels");
                    let mut labels = vec![0u8; n_labels];

                    match reader.read_exact(&mut labels) {
                        Err(e) => Err(format!("Could not read labels: {}", e)),
                        Ok(_) => Ok(labels)
                    }
                }
            }
        }
    }
}

pub fn load_images(path: &str) -> Result<Vec<Vec<u8>>, String> {
    match OpenOptions::new().read(true).open(path) {
        Err(e) => Err(format!("Could not open file {}: {}", path, e)),
        Ok(file) => {
            let mut reader = BufReader::new(file);

            match reader.seek(std::io::SeekFrom::Start(2)) {
                Err(e) => Err(format!("Could not seek in file {}: {}", path, e)),
                Ok(_) => {
                    let encoding = read_bytes!(reader, 1, "the data encoding byte");
                    if encoding != 0x08 {
                        return Err(format!("Unexpected encoding {}", encoding));
                    }

                    let dimensions = read_bytes!(reader, 1, "the data dimensions byte");
                    if dimensions != 0x03 {
                        return Err(format!("Unexpected dimensions {}", dimensions));
                    }

                    let n_images = read_bytes!(reader, 4, "the number of images");
                    let image_width = read_bytes!(reader, 4, "the width of an image");
                    let image_height = read_bytes!(reader, 4, "the height of an image");

                    let mut images = vec![vec![0u8; image_width * image_height]; n_images];

                    for img in images.iter_mut() {
                        match reader.read_exact(img) {
                            Err(e) => {
                                return Err(format!("Could not read image: {}", e));
                            },
                            Ok(_) => {}
                        }
                    }

                    Ok(images)
                }
            }
        }
    }
}

pub fn read_images_and_labels(images_path: &str, labels_path: &str) -> Result<Vec<Image>, String> {
    match (load_images(images_path), load_labels(labels_path)) {
        (Ok(images), Ok(labels)) => {
            if images.len() != labels.len() {
                Err(format!("Number of images ({}) does not match number of labels ({})", images.len(), labels.len()))
            } else {
                Ok(images
                    .into_iter()
                    .zip(labels.into_iter())
                    .map(
                        |(img, label)| Image {
                            pixels: img,
                            label: label
                        }
                ).collect())
            }
        },
        (Err(e), _) => Err(format!("Failed to load the images: {}", e)),
        (_, Err(e)) => Err(format!("Failed to load the labels: {}", e))
    }
}

pub fn load_training_set() -> Result<Vec<Image>, String> {
    read_images_and_labels(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte"
    )
}

pub fn load_testing_set() -> Result<Vec<Image>, String> {
    read_images_and_labels(
        "mnist/t10k-images.idx3-ubyte",
        "mnist/t10k-labels.idx1-ubyte"
    )
}

pub fn train() {
    let mut network = Network::new();

    network
        .add_layer(
            28 * 28,
            None,
            None
        )
        .add_layer(
            32,
            Some(NeuronActivation::LeakyReLU(0.01)),
            None
        )
        .add_layer(
            10,
            None,
            Some(LayerActivation::SoftMax),
        )
    ;

    let training = load_testing_set().unwrap();

    for image in training.iter() {
        println!("training {}", image.label);
        network.feed_forward(image);
    }
}
