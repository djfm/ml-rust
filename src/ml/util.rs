use std::{
    time::Duration,
};

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
    window_size: usize,
    data: &'a [T],
}

pub fn windows<'a, T>(slice: &'a [T], size: usize) -> WindowIterator<'a, T> {
    WindowIterator {
        current_index: 0,
        window_size: size,
        data: slice,
    }
}

impl<'a, T> Iterator for WindowIterator<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        let window_size = std::cmp::min(self.window_size, self.data.len() - self.current_index);

        let res = if window_size == 0 {
            None
        } else {
            Some(&self.data[self.current_index .. self.current_index + window_size])
        };

        self.current_index += window_size;

        res
    }
}

#[test]
fn test_windows() {
    let data = vec![1, 2, 3, 4];
    let mut iter = windows(&data, 3);
    assert_eq!(iter.next(), Some(&data[0..3]));
    assert_eq!(iter.next(), Some(&data[3..]));
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
