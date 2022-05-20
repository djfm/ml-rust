use std::{
    time::{Duration, Instant},
    cell::RefCell,
};

pub struct Timer {
    description: String,
    t_start: Instant,
}

impl Timer {
    pub fn start(description: &str) -> Self {
        println!("Starting timer: {}", description);

        Timer {
            description: description.to_string(),
            t_start: Instant::now(),
        }
    }

    pub fn stop(self) -> Duration {
        let elapsed = self.t_start.elapsed();
        println!("Timer stopped: {} done in {}", self.description, human_duration(elapsed));
        elapsed
    }
}

pub fn human_duration(duration: Duration) -> String {
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

pub struct WindowIterator<'a, T> {
    current_index: usize,
    conf: &'a WindowIteratorConfig,
    data: &'a [T],
}

impl <'a, T> WindowIterator<'a, T> {
    pub fn new(data: &'a [T], conf: &'a WindowIteratorConfig) -> Self {
        Self {
            current_index: 0,
            conf,
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.conf.get_size()
    }
}

pub struct WindowIteratorConfig {
    pub size: RefCell<usize>,
}

impl WindowIteratorConfig {
    pub fn new(size: usize) -> Self {
        Self {
            size: RefCell::new(size),
        }
    }

    fn get_size(&self) -> usize {
        *self.size.borrow()
    }

    pub fn set_size(&self, size: usize) {
        *self.size.borrow_mut() = size;
    }
}

pub fn windows<'a, T>(slice: &'a [T], conf: &'a WindowIteratorConfig) -> WindowIterator<'a, T>
{
    WindowIterator::new(slice, conf)
}

impl<'a, T> Iterator for WindowIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let window_size = std::cmp::min(
            self.size(),
            self.data.len() - self.current_index
        );

        let res = if window_size == 0 {
            None
        } else {
            Some(&self.data[self.current_index .. self.current_index + window_size])
        };

        self.current_index += window_size;

        res
    }
}

pub fn average_scalar_vectors(sources: &[Vec<f32>]) -> Vec<f32> {
    if sources.is_empty() {
        return vec![];
    }

    let len = sources.first().expect("there should be a first vector").len();
    let mut result = vec![0.0; len];

    for i in 0..len {
        for s in sources {
            result[i] += s[i] / len as f32;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows() {
        let data = vec![1, 2, 3, 4];
        let conf = WindowIteratorConfig::new(2);
        let mut iter = windows(&data, &conf);
        assert_eq!(iter.next(), Some(&data[0..2]));
        assert_eq!(iter.next(), Some(&data[2..]));
    }

    #[test]
    fn test_decreasing_windows() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let conf = WindowIteratorConfig::new(3);
        let mut iter = windows(&data, &conf);
        assert_eq!(iter.next(), Some(&data[0..3]));
        conf.set_size(4);
        assert_eq!(iter.next(), Some(&data[3..7]));
    }

    #[test]
    fn test_human_duration() {
        assert_eq!(human_duration(Duration::new(0, 0)), "0s");
        assert_eq!(human_duration(Duration::new(61, 0)), "1m 1s");
        assert_eq!(human_duration(Duration::new(3610, 0)), "1h 10s");
    }

    #[test]
    fn test_ref() {
        let vi: Vec<bool> = Vec::new();
        let mut batches = vec![vi; 4];
        batches[0] = vec![false];
        batches[1] = vec![true, true];
        assert_ne!(batches[0], batches[1]);
    }

    #[test]
    fn test_average_vectors() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let avg = average_scalar_vectors(&vectors);
        let expected = vec![
            4.0, 5.0, 6.0,
        ];

        assert_eq!(avg, expected);
    }

    #[test]
    #[should_panic]
    fn test_sum_vectors_fail() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
        ];

        average_scalar_vectors(&vectors);
    }
}
