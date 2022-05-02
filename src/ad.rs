use std::ops;

pub struct ADNumber {
    id: usize,
    scalar: f32,
    tape: Tape,
    gradient: Option<Vec<f32>>,
}

struct Tape {
    records: Vec<TapeRecord>,
}

#[derive(Clone)]
struct TapeRecord {
    partials: Vec<PartialDiff>,
}

#[derive(Clone)]
struct PartialDiff {
    with_respect_to_id: usize,
    partial_derivative: f32,
}

impl Tape {
    fn new() -> Tape {
        Tape {
            records: Vec::new(),
        }
    }
}

impl ADNumber {
    pub fn new(scalar: f32) -> ADNumber {
        ADNumber {
            id: 0,
            scalar,
            tape: Tape::new(),
            gradient: None,
        }
    }

    fn merge(a: ADNumber, b: ADNumber) -> ADNumber {
        let mut tape = Tape {
            records: [a.tape.records, b.tape.records].concat(),
        };

        ADNumber {
            id: tape.records.len(),
            scalar: a.scalar + b.scalar,
            tape,
            gradient: None,
        }
    }

    fn compute_gradient(&mut self, y: &ADNumber) -> Vec<f32> {
        let y_id = y.id;
        let mut dy = vec![0.0; y_id + 1];

        dy[y_id] = 1.0;
        for i in (0..y_id+1).rev() {
            match self.tape.records.get(i) {
                Some(record) => {
                    for partial in &record.partials {
                        dy[partial.with_respect_to_id] += partial.partial_derivative * dy[i];
                    }
                },
                None => {
                    panic!("partial derivative of expression not found on tape");
                },
            }
        }
        dy
    }

    pub fn diff(&mut self, wrt: ADNumber) -> f32 {
        if self.gradient.is_none() {
            let gradient = self.compute_gradient(&wrt);
            self.gradient = Some(gradient);
        }

        match self.gradient {
            Some(ref gradient) => {
                match gradient.get(wrt.id) {
                    Some(&value) => value,
                    None => panic!("cannot differentiate with respect to that expression"),
                }
            },
            None => {
                panic!("gradient of expression not found");
            },
        }
    }
}

impl ops::Add<ADNumber> for ADNumber {
    type Output = ADNumber;

    fn add(self, other: ADNumber) -> ADNumber {
        ADNumber::merge(self, other)
    }
}

impl ops::Add<f32> for ADNumber {
    type Output = ADNumber;

    fn add(self, other: f32) -> ADNumber {
        self.add(ADNumber::new(other))
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = ADNumber::new(1.0);
        let b = ADNumber::new(2.0);
        let c = a + b;
        assert_eq!(c.scalar, 3.0);
    }
}
