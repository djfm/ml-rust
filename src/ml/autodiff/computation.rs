use std::ops;

pub enum ComputationRecord {
    Add(ADNum, ADNum),
    Sub(ADNum, ADNum),
}

use crate::ml::NumberLike;

type Computation = Fn(&mut ComputationEnvironment) -> ();

pub struct ComputationEnvironment {

}

#[derive(Copy, Clone)]
pub struct ADNum {
    id: Option<usize>,
    scalar: f32,
}

impl NumberLike for ADNum {
    fn scalar(&self) -> f32 {
        return self.scalar;
    }
}

/*
impl ops::Add<ADNum> for ADNum {
    type Output = ADNum;

    fn add(self, other: ADNum) -> ADNum {

    }
}
*/

pub struct ADNumFactory {
    max_id: usize,
}

impl ADNumFactory {
    pub fn new() { Default::default() }

    fn next_id(&mut self) -> usize {
        let id = self.max_id;
        self.max_id += 1;
        id
    }

    pub fn constant(&mut self, scalar: f32) -> ADNum {
        ADNum { scalar, id: None }
    }

    pub fn variable(&mut self, scalar: f32) -> ADNum {
        ADNum { scalar, id: Some(self.next_id()) }
    }
}

impl ComputationEnvironment {
    pub fn new() -> Self {
        Self {

        }
    }

    pub fn compute(&mut self, computation: &Computation) {
        computation(self);
    }
}
