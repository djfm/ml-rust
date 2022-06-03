use rand::thread_rng;
use rand::seq::SliceRandom;

use crate::{
    ml::{
        Network,
        ClassificationExample,
        AutoDiff,
        FloatFactory,
        BatchResult,
    },
    util::{
        windows,
        Timer,
        WindowIteratorConfig,
    }
};

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    epochs: usize,
    learning_rate: f32,
    learning_rate_decay: f32,
    batch_size: usize,
    batch_size_fp: f32,
    batch_size_decay: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            learning_rate: 0.001,
            learning_rate_decay: 0.001,
            batch_size: 50,
            batch_size_fp: 50.0,
            batch_size_decay: 0.001,
        }
    }
}

impl TrainingConfig {
    pub fn new(epochs: usize, learning_rate: f32, batch_size: usize) -> Self {
        Self {
            epochs,
            learning_rate,
            batch_size,
            batch_size_fp: batch_size as f32,
            ..Default::default()
        }
    }

    pub fn update(&mut self, batch_result: &BatchResult) -> &mut Self {
        self.learning_rate -= self.learning_rate * 0.001 / batch_result.batch_size() as f32;
        self.batch_size_fp -= self.batch_size_fp * 0.001 / batch_result.batch_size() as f32;
        self.batch_size = self.batch_size_fp.round() as usize;
        self
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}



pub fn train<S: ClassificationExample>(
    network: &mut Network,
    training_set: &[S],
    testing_set: &[S],
    training_config: TrainingConfig,
) {
    let tconf = &mut training_config.clone();
    let timer = Timer::start(&format!("training on {} samples", training_set.len()));
    let nf_creator = || AutoDiff::new();

    let win_iter_conf = WindowIteratorConfig::new(tconf.batch_size);

    let mut processed = 0;
    let total = training_set.len() * tconf.epochs;

    let mut tset = training_set.to_vec();

    for epoch in 1..=tconf.epochs {
        for batch in windows(&tset, &win_iter_conf) {
            let batch_result = network.feed_batch_forward(nf_creator, batch);

            processed += batch.len();
            let progress = 100.0 * processed as f32 / total as f32;

            network.back_propagate(&batch_result.diffs(), &tconf);

            println!(
                "Epoch {}/{}, {} samples ({:.2}%) processed. Batch accuracy is: {:.2}%\n",
                epoch, tconf.epochs, processed, progress, batch_result.accuracy(),
            );

            tconf.update(&batch_result);

            println!("Updated training params: {:?}", tconf);
        }

        println!("\nEpoch {}/{} finished. Testing...", epoch, tconf.epochs);
        let ff_provider = || FloatFactory::new();
        let error = network.feed_batch_forward(ff_provider, testing_set);
        println!("Testing finished. Accuracy is: {:.2}%\n", error.accuracy());

        tset.shuffle(&mut thread_rng());
    }

    timer.stop();
}
