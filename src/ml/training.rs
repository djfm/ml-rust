use crate::{
    ml::{
        Network,
        ClassificationExample,
        AutoDiff,
    },
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
            learning_rate: 0.01,
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

}
