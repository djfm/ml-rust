use crate::ml::{
    NumberLike,
};

#[derive(Clone, Copy, Debug)]
pub struct ADNumber {
    id: usize,
    scalar: f32,
}

impl ADNumber {
    pub fn new(id: usize, scalar: f32) -> Self {
        Self {
            id, scalar
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn scalar(&self) -> f32 {
        self.scalar
    }
}

impl NumberLike for ADNumber {

}
