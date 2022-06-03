use rand::thread_rng;
use rand::seq::SliceRandom;
use crossbeam_utils::thread;
use crossbeam_channel::{
    unbounded,
    Sender,
    Receiver,
};

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
    },
    plotter,
};

#[derive(Copy, Clone, Debug)]
enum AccuracyDataPoint {
    Batch (f32, f32),
    Epoch (f32, f32),
}

impl plotter::DataPoint for AccuracyDataPoint {
    fn x(&self) -> f32 {
        match self {
            AccuracyDataPoint::Batch (x, _) => *x,
            AccuracyDataPoint::Epoch (x, _) => *x,
        }
    }

    fn y(&self) -> f32 {
        match self {
            AccuracyDataPoint::Batch (_, y) => *y,
            AccuracyDataPoint::Epoch (_, y) => *y,
        }
    }

    fn series_name(&self) -> &str {
        match self {
            AccuracyDataPoint::Batch (_, _) => "Batch Accuracy",
            AccuracyDataPoint::Epoch (_, _) => "Epoch Accuracy",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    epochs: usize,
    training_samples_count: usize,
    training_samples_seen: usize,
    initial_batch_size: usize,
    initial_learning_rate: f32,
    learning_rate: f32,
    target_learning_rate: f32,
    batch_size: usize,
    target_batch_size: usize,
    progress: f32,
}

impl TrainingConfig {
    pub fn new(
        epochs: usize,
        training_set_size: usize,
        learning_rate: f32,
        target_learning_rate: f32,
        batch_size: usize,
        target_batch_size: usize
    ) -> Self {
        Self {
            epochs,
            learning_rate,
            target_learning_rate,
            batch_size,
            target_batch_size,
            progress: 0.0,
            training_samples_count: epochs * training_set_size,
            training_samples_seen: 0,
            initial_batch_size: batch_size,
            initial_learning_rate: learning_rate,
        }
    }

    pub fn update(&mut self, batch_result: &BatchResult) -> &mut Self {
        self.training_samples_seen += batch_result.batch_size();
        self.progress = self.training_samples_seen as f32 / self.training_samples_count as f32;

        self.learning_rate =
            self.initial_learning_rate * (1.0 - self.progress) +
            self.target_learning_rate * self.progress;

        self.batch_size = (
            self.initial_batch_size as f32 * (1.0 - self.progress) +
            self.target_batch_size as f32 * self.progress
        ).round() as usize;

        self
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}



fn do_train<'a, S: ClassificationExample>(
    network: &'a mut Network,
    training_set: &[S],
    testing_set: &[S],
    training_config: TrainingConfig,
    send: &mut Sender<AccuracyDataPoint>,
) -> &'a mut Network {
    let t_conf = &mut training_config.clone();
    let timer = Timer::start(&format!("training on {} samples", training_set.len()));
    let nf_creator = || AutoDiff::new();

    let win_iter_conf = WindowIteratorConfig::new(t_conf.batch_size);

    let mut processed = 0;
    let total = training_set.len() * t_conf.epochs;

    let mut t_set = training_set.to_vec();

    for epoch in 1..=t_conf.epochs {
        for batch in windows(&t_set, &win_iter_conf) {
            let batch_result = network.feed_batch_forward(nf_creator, batch);

            processed += batch.len();
            let progress = 100.0 * processed as f32 / total as f32;
            let point = AccuracyDataPoint::Batch(progress, batch_result.accuracy());

            if let Err(error) = send.send(point) {
                println!("Error sending batch data point {}: ", error);
            }

            network.back_propagate(&batch_result.diffs(), &t_conf);

            println!(
                "Epoch {}/{}, {} samples ({:.2}%) processed. Batch accuracy is: {:.2}%",
                epoch, t_conf.epochs, processed, progress, batch_result.accuracy(),
            );

            t_conf.update(&batch_result);

            println!("Updated training params: {:#?}\n", t_conf);
        }

        println!("\nEpoch {}/{} finished. Testing...", epoch, t_conf.epochs);
        let ff_provider = || FloatFactory::new();
        let error = network.feed_batch_forward(ff_provider, testing_set);
        println!("Testing finished. Accuracy is: {:.2}%\n", error.accuracy());

        if let Err(error) = send.send(AccuracyDataPoint::Epoch(
            epoch as f32,
            error.accuracy(),
        )) {
            println!("Error sending epoch data point {}: ", error);
        }

        t_set.shuffle(&mut thread_rng());
    }

    timer.stop();
    network
}



pub fn train<'a, S: ClassificationExample>(
    network: &'a mut Network,
    training_set: &'a [S],
    testing_set: &'a [S],
    training_config: TrainingConfig,
) -> &'a mut Network {
    let (mut sender, mut receiver): (Sender<AccuracyDataPoint>, Receiver<AccuracyDataPoint>) = unbounded();

    let handles = thread::scope(|s| {
        s.spawn(|_| {
            do_train(
                network,
                training_set, testing_set,
                training_config, &mut sender,
            )
        });

        s.spawn(|_| {
            plotter::plot(&mut receiver);
        });
    });

    match handles {
        Ok(_) => {
            network
        },
        Err(e) => panic!("training failed {:#?}", e),
    }
}
