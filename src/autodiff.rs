use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
pub struct ADValue {
  id: usize,
  value: f32,
}

impl ADValue {
  pub fn scalar(&self) -> f32 {
    self.value
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

  pub fn create_variable(&mut self, value: f32) -> ADValue {
    let id = self.tape.len();
    self.tape.records.push(TapeRecord {
      partials: Vec::new(),
    });
    ADValue {
      id,
      value,
    }
  }

  fn compute_gradient(&mut self, y: &ADValue) {
    let mut dy = vec![0.0; y.id + 1];

    dy[y.id] = 1.0;
    for i in (0..y.id+1).rev() {
      for record in &self.tape.records[i].partials {
        dy[record.with_respect_to_id] += dy[i] * record.value;
      }
    }

    self.gradients.insert(y.id, dy);
  }

  pub fn diff(&mut self, y: &ADValue, wrt: &ADValue) -> f32 {
    match self.gradients.get(&y.id) {
      Some(dy) => dy[wrt.id],
      None => {
        self.compute_gradient(y);
        self.gradients[&y.id][wrt.id]
      }
    }
  }

  pub fn add(&mut self, values: &Vec<ADValue>) -> ADValue {
    let id = self.tape.len();

    let partials = values.iter().map(|value| {
      PartialDiff {
        with_respect_to_id: value.id,
        value: 1.0,
      }
    }).collect();

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id,
      value: values.iter().map(|value| value.scalar()).sum(),
    }
  }

  pub fn mul(&mut self, values: &Vec<ADValue>) -> ADValue {
    let id = self.tape.len();

    let partials = values.iter().enumerate().map(|(i, value)| {
      PartialDiff {
        with_respect_to_id: value.id,
        value: values.iter().enumerate().map(|(j, value)| {
          if i == j {
            1.0
          } else {
            value.scalar()
          }
        }).product(),
      }
    }).collect();

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id,
      value: values.iter().map(|value| value.scalar()).product(),
    }
  }

  pub fn div2(&mut self, left: &ADValue, right: &ADValue) -> ADValue {
    let id = self.tape.len();

    let partials = vec![
      PartialDiff {
        with_respect_to_id: left.id,
        value: 1.0 / right.scalar(),
      },
      PartialDiff {
        with_respect_to_id: right.id,
        value: -left.scalar() / right.scalar().powi(2),
      },
    ];

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id,
      value: left.scalar() / right.scalar(),
    }
  }

  pub fn sub(&mut self, values: &Vec<ADValue>) -> ADValue {
    let id = self.tape.len();

    let partials = values.iter().enumerate().map(|(i, value)| {
        PartialDiff {
          with_respect_to_id: value.id,
          value: if i == 0 { 1.0 } else { -1.0 },
        }
    }).collect();

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id,
      value: values.iter().enumerate().map(
        |(i, value)| if i == 0 { value.scalar() } else { -value.scalar() }
      ).sum(),
    }
  }

  pub fn exp(&mut self, value: &ADValue) -> ADValue {
    let exp = value.scalar().exp();

    let id = self.tape.len();

    let partials = vec![
      PartialDiff {
        with_respect_to_id: value.id,
        value: exp,
      }
    ];

    self.tape.records.push(TapeRecord {
      partials,
    });

    ADValue {
      id,
      value: exp,
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
    let dy_dx = ad.diff(&y, &x);
    assert_eq!(y.scalar(), 2.0);
    assert_eq!(dy_dx, 2.0);
  }

  #[test]
  fn test_dx2_dx() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(2.0);
    let y = ad.mul(&vec![x, x]);
    let dy_dx = ad.diff(&y, &x);
    assert_eq!(dy_dx, 4.0);
    assert_eq!(y.scalar(), 4.0);
  }

  #[test]
  fn test_dx2y_dx_dx2y_dy() {
      let mut ad = AutoDiff::new();
      let x = ad.create_variable(2.0);
      let y = ad.create_variable(3.0);
      let z = ad.mul(&vec![x, x, y]);

      assert_eq!(ad.diff(&z, &x), 12.0);
      assert_eq!(ad.diff(&z, &y), 4.0);
  }

  #[test]
  fn test_sub() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(1.0);
    let y = ad.create_variable(2.0);
    let z = ad.sub(&vec![x, y]);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);
    assert_eq!(dz_dx, 1.0);
    assert_eq!(dz_dy, -1.0);
    assert_eq!(z.scalar(), -1.0);
  }

  #[test]
  fn test_mul() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(3.0);
    let y = ad.create_variable(2.0);
    let z = ad.mul(&vec![x, y]);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);
    assert_eq!(dz_dx, 2.0);
    assert_eq!(dz_dy, 3.0);
    assert_eq!(z.scalar(), 6.0);
  }

  #[test]
  fn test_div() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(1.0);
    let y = ad.create_variable(2.0);
    let z = ad.div2(&x, &y);
    let dz_dx = ad.diff(&z, &x);
    let dz_dy = ad.diff(&z, &y);
    assert_eq!(dz_dx, 0.5);
    assert_eq!(dz_dy, -0.25);
    assert_eq!(z.scalar(), 0.5);
  }

  #[test]
  fn test_exp() {
    let mut ad = AutoDiff::new();
    let x = ad.create_variable(1.0);
    let y = ad.exp(&x);
    let dy_dx = ad.diff(&y, &x);
    assert_eq!(dy_dx, 1.0f32.exp());
  }

  #[test]
  fn test_much_more_complex_diff() {
      let mut ad = AutoDiff::new();
      let x = ad.create_variable(3.0);
      let y = ad.create_variable(4.0);
      let exp_x = ad.exp(&x);
      let exp_x_minus_y = ad.sub(&vec![exp_x, y]);
      let o = ad.div2(&y, &exp_x_minus_y);
      print!("{:#?}", ad);

      assert_eq!(ad.diff(&o, &x), -0.310507656);
      assert_eq!(ad.diff(&o, &y), 0.077626914);
  }
}