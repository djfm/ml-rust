use crate::ml::{
    ADNumber,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CellActivation {
    None,
    LeakyReLU(f32),
    ReLu,
}

impl CellActivation {
    pub fn compute<'a>(&'a self, x: &'a ADNumber<'a>) -> ADNumber<'a> {
        match self {
            CellActivation::None => x.clone(),
            CellActivation::LeakyReLU(alpha) => {
                if x.scalar() < 0.0 {
                    x.ad().create_constant(*alpha)
                } else {
                    x.clone()
                }
            }
            CellActivation::ReLu => {
                if x.scalar() < 0.0 {
                    x.ad().create_constant(0.0)
                } else {
                    x.clone()
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LayerActivation {
    None,
    SoftMax,
}

impl LayerActivation {
    pub fn compute<'a>(&'a self, vec: &'a Vec<ADNumber<'a>>) -> Vec<ADNumber<'a>> {
        if vec.is_empty() {
            return vec![];
        }

        let ad = vec[0].ad();

        match self {
            LayerActivation::None => vec.clone(),
            LayerActivation::SoftMax => {
                let mut sum = ad.create_constant(0.0);
                let mut res = vec![ad.create_constant(0.0); vec.len()];

                for (i, x) in vec.iter().enumerate() {
                    let exp = x.exp();
                    sum = sum + exp;
                    res[i] = exp;
                }

                for x in res.iter_mut() {
                    *x = *x / sum;
                }

                res
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ErrorFunction {
    EuclideanDistanceSquared,
}

impl ErrorFunction {
    pub fn compute<'a>(&'a self, x: &'a Vec<ADNumber<'a>>, y: &'a Vec<ADNumber<'a>>) -> ADNumber<'a> {
        if x.len() != y.len() {
            panic!("x and y must have the same length");
        }

        if x.is_empty() {
            panic!("x and y must not be empty");
        }

        let ad = x[0].ad();

        match self {
            ErrorFunction::EuclideanDistanceSquared => {
                let mut sum = ad.create_constant(0.0);

                for (x, y) in x.iter().zip(y.iter()) {
                    let diff = *x - *y;
                    sum = sum + diff * diff;
                }

                sum
            }
        }
    }
}

pub fn one_hot_label(input: &Vec<ADNumber>) -> usize {
    let mut max = input[0];
    let mut max_index = 0;

    for (i, n) in input.iter().enumerate() {
        if n > &max {
            max = *n;
            max_index = i;
        }
    }

    max_index
}
