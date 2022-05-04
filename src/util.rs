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
