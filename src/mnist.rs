use std::fs::OpenOptions;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::time::{Instant, Duration};
use crossbeam_utils::thread;

use super::ff_network::{
    BatchResult,
    Network,
    ClassificationExample,
};

use super::activations::{
    NeuronActivation,
    LayerActivation,
};

use super::graphics::{
    show_mnist_image,
};

fn human_duration(duration: Duration) -> String {
    let mut remaining_secs = duration.as_secs();
    let decomposition_factors = [
        (20 * 3600 * 30 * 12, "y"),
        (24 * 3600 * 30, "M"),
        (24 * 3600, "d"),
        (3600, "h"),
        (60, "m"),
        (1, "s"),
    ];
    let mut result_parts = Vec::new();

    for &(divisor, suffix) in decomposition_factors.iter() {
        if remaining_secs >= divisor {
            let num = remaining_secs / divisor;
            remaining_secs %= divisor;
            result_parts.push(format!("{}{}", num, suffix));
        }
    }

    if result_parts.len() == 0 {
        result_parts.push("0s".to_string());
    }

    result_parts.join(" ")
}

#[derive(Hash)]
pub struct Image {
    pub pixels: Vec<u8>,
    pub label: u8,
}

impl Image {
    pub fn calculate_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl ClassificationExample for Image {
    fn get_input(&self) -> Vec<f32> {
        self.pixels.iter().map(|&x| x as f32 / 255.0).collect()
    }

    fn get_label(&self) -> usize {
        self.label as usize
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

pub fn create_network() -> Network {
    let mut network = Network::new();

    network
        .add_layer(
            28 * 28,
            false,
            None,
            None
        )
        .add_layer(
            32,
            true,
            Some(NeuronActivation::LeakyReLU(0.01)),
            None
        )
        .add_layer(
            10,
            false,
            None,
            Some(LayerActivation::SoftMax),
        )
    ;

    network
}

pub fn train() {
    let start_instant = Instant::now();
    let mut network = create_network();

    println!(
        "Created a network with {} layers totalling {} parameters",
        network.depth(),
        network.autodiff().size()
    );

    let training = load_training_set().unwrap();
    println!("Loaded {} MNIST training samples", training.len());
    let testing = load_testing_set().unwrap();
    println!("Loaded {} MNIST testing samples", testing.len());

    // Training params
    let batch_size = 32;
    let learning_rate = 0.01;
    let epochs = 5;

    // Utility variables
    let mut batch_error = network.autodiff().create_variable(0.0);
    let mut current_batch_size = 0;
    let mut batch_number = 0;

    for epoch in 1..=epochs {
        for (i, image) in training.iter().enumerate() {
            let error = network.compute_example_error(image);
            batch_error = network.autodiff().add(batch_error, error);
            current_batch_size += 1;

            if current_batch_size >= batch_size || i >= training.len() - 1 {
                let size = network.autodiff().create_constant(current_batch_size as f32);
                batch_error = network.autodiff().div(batch_error, size);
                network.back_propagate(batch_error, learning_rate);

                println!(
                    "batch {} of epoch {} trained, error: {:.2}%",
                    batch_number,
                    epoch,
                    100.0 * batch_error.value,
                );

                batch_error = network.autodiff().create_variable(0.0);
                current_batch_size = 0;
                batch_number += 1;
            }
        }

        println!("Training of epoch {} done, now testing...", epoch);
        let testing_accuracy = network.compute_accuracy(&testing);
        println!("Accuracy on testing set: {:.2}%", testing_accuracy);

        batch_number = 0;
    }

    println!("Training complete! (in {})", human_duration(start_instant.elapsed()));
}

pub fn train_parallel() {
    let start_instant = Instant::now();
    let mut network = create_network();

    println!(
        "Created a network with {} layers totalling {} parameters",
        network.depth(),
        network.autodiff().size()
    );

    let training = load_training_set().unwrap();
    println!("Loaded {} MNIST training samples", training.len());
    let testing = load_testing_set().unwrap();
    println!("Loaded {} MNIST testing samples", testing.len());

    // Training params
    let batch_size = 32;
    let learning_rate = 0.01;
    let epochs = 5;
    let n_threads = num_cpus::get();

    // Utility variables
    let mut image_pos = 0;
    let mut epoch = 0;

    while epoch < epochs {
        thread::scope(|s| {
            let mut thread_handles = Vec::new();

            for _ in 0..n_threads {
                let batch: Vec<&Image> = training.iter().skip(image_pos).take(batch_size).collect();

                image_pos += batch.len();

                let mut res = BatchResult {
                    error: network.autodiff().create_variable(0.0),
                    net: network.clone(),
                };

                thread_handles.push(
                    s.spawn(move |_| {
                        res.error = res.net.compute_batch_error(&batch);
                        res
                    })
                );
            }

            let mut networks = Vec::new();

            for h in thread_handles {
                match h.join() {
                    Ok(BatchResult { error, mut net }) => {
                        net.back_propagate(error, learning_rate);
                        networks.push(net);
                        println!("Batch error: {:.2}%", 100.0 * error.value);
                    },
                    Err(e) => {
                        println!("Could not compute batch error: {:?}", e);
                    }
                }
            }

            network.average(&networks);
        }).unwrap();

        println!("trained on {} images so far...", image_pos);

        if image_pos >= training.len() - 1 {
            println!("Training of epoch {} done, now testing...", epoch);
            let testing_accuracy = network.compute_accuracy(&testing);
            println!("Accuracy on testing set: {:.2}%", testing_accuracy);
            image_pos = 0;
            epoch += 1;
        }
    }

    println!("Training complete! (in {})", human_duration(start_instant.elapsed()));
}

mod tests {
    use super::*;

    #[test]
    fn test_human_duration() {
        assert_eq!(human_duration(Duration::new(0, 0)), "0s");
        assert_eq!(human_duration(Duration::new(61, 0)), "1m 1s");
        assert_eq!(human_duration(Duration::new(3610, 0)), "1h 10s");
    }

    fn test_ref() {
        let vi: Vec<bool> = Vec::new();
        let mut batches = vec![vi; 4];
        batches[0] = vec![false];
        batches[1] = vec![true, true];
        assert_ne!(batches[0], batches[1]);
    }
}
