#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub learning_rate_decay: f32,
    pub min_learning_rate: f32,
    pub epochs: usize,
    pub batch_size: f32,
    pub batch_size_decay: f32,
    pub min_batch_size: f32,
    pub training_samples_len: usize,
}

impl TrainingConfig {
    pub fn new(training_samples_len: usize) -> Self {
        Self {
            learning_rate: 0.01,
            learning_rate_decay: 0.9999,
            min_learning_rate: 0.00001,
            epochs: 5,
            batch_size: 80.0,
            batch_size_decay: 0.9999,
            min_batch_size: 30.0,
            training_samples_len,
        }
    }

    pub fn update(&mut self, batch_size: usize) {
        let current_batch_size = self.batch_size;
        let decay = |x: &mut f32, decay: f32, min: f32| {
            *x = (*x * decay.powf(batch_size as f32 / current_batch_size)).max(min);
        };

        decay(&mut self.learning_rate, self.learning_rate_decay, self.min_learning_rate);
        decay(&mut self.batch_size, self.batch_size_decay, self.min_batch_size);
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size as usize
    }
}
