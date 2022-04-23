use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
pub struct ADValue {
  id: Option<usize>,
  value: f32,
}

impl ADValue {
  pub fn is_constant(&self) -> bool {
    self.id.is_none()
  }

  pub fn is_variable(&self) -> bool {
    self.id.is_some()
  }
}

#[derive(Debug)]
struct PartialDiff {
  with_respect_to_id: usize,
  value: f32,
}

#[derive(Debug)]
struct TapeRecord {
  partials: Vec<PartialDiff>,
}

#[derive(Debug)]
struct Tape {
  records: Vec<TapeRecord>,
}

impl Tape {
  pub fn new() -> Tape {
    Tape {
      records: Vec::new(),
    }
  }

  pub fn len(&self) -> usize {
    self.records.len()
  }
}

#[derive(Debug)]
pub struct AutoDiff {
  tape: Tape,
  gradients: HashMap<usize, Vec<f32>>,
}

impl AutoDiff {
  pub fn new() -> AutoDiff {
    AutoDiff {
      tape: Tape::new(),
      gradients: HashMap::new(),
    }
  }

  pub fn create_constant(&mut self, value: f32) -> ADValue {
    ADValue {
      id: None,
      value,
    }
  }

  pub fn create_variable(&mut self, value: f32) -> ADValue {
    let id = self.tape.len();
    self.tape.records.push(TapeRecord {
      partials: Vec::new(),
    });
    ADValue {
      id: Some(id),
      value,
    }
  }

  fn compute_gradient(&mut self, y: &ADValue) {
    let y_id = y.id.unwrap();
    let mut dy = vec![0.0; y_id + 1];

    dy[y_id] = 1.0;
    for i in (0..y_id+1).rev() {
      for record in &self.tape.records[i].partials {
        dy[record.with_respect_to_id] += dy[i] * record.value;
      }
    }

    self.gradients.insert(y_id, dy);
  }

  pub fn diff(&mut self, y: &ADValue, wrt: &ADValue) -> f32 {
    if y.is_constant() {
      return 0.0;
    }

    if (wrt.is_constant()) {
      panic!("cannot differentiate with respect to a constant");
    }

    match self.gradients.get(&y.id.unwrap()) {
      Some(dy) => dy[wrt.id.unwrap()],
      None => {
        self.compute_gradient(y);
        self.gradients[&y.id.unwrap()][wrt.id.unwrap()]
      }
    }
  }

  pub fn add(&mut self, values: &Vec<ADValue>) -> ADValue {
    let id = self.tape.len();

    let mut partials = values.iter().map(|value| {
      PartialDiff {
        with_respect_to_id: value.id.unwrap(),
        value: 1.0,
      }
    }).collect();

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id: Some(id),
      value: values.iter().map(|value| value.value).sum(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add_simple() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(1.0);
    let y = ad.add(&vec![x, x]);

    print!("{:#?}", ad);

    let dy_dx = ad.diff(&y, &x);
    assert_eq!(dy_dx, 2.0);
  }
}