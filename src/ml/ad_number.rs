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

impl std::cmp::Eq for ADNumber {
}

impl std::cmp::PartialEq for ADNumber {
    fn eq(&self, other: &Self) -> bool {
        self.scalar == other.scalar
    }
}

impl std::cmp::PartialOrd for ADNumber {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.scalar.partial_cmp(&other.scalar)
    }
}

impl std::cmp::Ord for ADNumber {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.scalar < other.scalar {
            std::cmp::Ordering::Less
        } else if self.scalar > other.scalar {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl NumberLike for ADNumber {}
