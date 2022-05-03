use crate::ml::ad_number::{
    ADNumber,
};

impl <'a> std::default::Default for ADNumber<'a> {
    fn default() -> Self {
        Self::new(0, 0, None, 0.0)
    }
}

impl <'a> std::cmp::PartialEq for ADNumber<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.scalar == other.scalar
    }
}

impl <'a> std::cmp::PartialOrd for ADNumber<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.scalar.partial_cmp(&other.scalar)
    }
}
