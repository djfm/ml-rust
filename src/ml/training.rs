use crate::{
    ml::{
        Network,
        ClassificationExample,
        AutoDiff,
        FloatFactory,
    },
    util::{
        windows,
        Timer,
        WindowIteratorConfig,
    }
};

pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            learning_rate: 0.001,
            batch_size: 50,
        }
    }
}



pub fn train<S: ClassificationExample>(
    network: &mut Network,
    training_set: &[S],
    testing_set: &[S],
    tconf: &TrainingConfig,
) {
    let timer = Timer::start(&format!("training on {} samples", training_set.len()));
    let nf_creator = || AutoDiff::new();

    let win_iter_conf = WindowIteratorConfig::new(tconf.batch_size);

    let mut processed = 0;
    let total = training_set.len() * tconf.epochs;

    for epoch in 1..=tconf.epochs {
        for batch in windows(training_set, &win_iter_conf) {
            let batch_result = network.feed_batch_forward(nf_creator, batch);

            processed += batch.len();
            let progress = 100.0 * processed as f32 / total as f32;

            network.back_propagate(&batch_result.diffs(), &tconf);

            println!(
                "Epoch {}/{}, {} samples ({:.2}%) processed. Batch accuracy is: {}\n",
                epoch, tconf.epochs, processed, progress, batch_result.accuracy(),
            );
        }

        println!("\nEpoch {}/{} finished. Testing...", epoch, tconf.epochs);
        let ff_provider = || FloatFactory::new();
        let error = network.feed_batch_forward(ff_provider, testing_set);
        println!("Testing finished. Accuracy is: {}\n", error.accuracy());
    }

    timer.stop();
}

/*
pub fn train_classical<S: ClassificationExample>(
    network: &mut Network,
    training_set: &[S],
    testing_set: &[S],
    tconf: &TrainingConfig,
) {
    let timer = Timer::start(&format!("training on {} samples", training_set.len()));
    let nf_creator = || AutoDiff::new();

    let win_iter_conf = WindowIteratorConfig::new(tconf.batch_size);

    let mut processed = 0;
    let total = training_set.len() * tconf.epochs;

    for epoch in 1..=tconf.epochs {
        for batch in windows(training_set, &win_iter_conf) {
            let mut nf = AutoDiff::new();
            let error = batch.iter().fold(nf.from_scalar(0.0), |acc, sample| {
                let ff = network.feed_forward(&mut nf, sample);
                nf.add(&acc, ff.error)
            });

            processed += batch.len();
            let progress = 100.0 * processed as f32 / total as f32;

            network.back_propagate(&batch_result.diffs(), &tconf);

            println!(
                "Epoch {}/{}, {} samples ({:.2}%) processed. Batch accuracy is: {}\n",
                epoch, tconf.epochs, processed, progress, batch_result.accuracy(),
            );
        }

        println!("\nEpoch {}/{} finished. Testing...", epoch, tconf.epochs);
        let ff_provider = || FloatFactory::new();
        let error = network.feed_batch_forward(ff_provider, testing_set);
        println!("Testing finished. Accuracy is: {}\n", error.accuracy());
    }

    timer.stop();
}
*/
