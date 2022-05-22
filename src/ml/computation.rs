pub struct Computation {}
pub struct ComputationHelper {}

impl Computation {
    pub fn new() -> Computation {
        Self {
        }
    }

    pub fn make_compute_helper(&mut self) -> ComputationHelper {
        ComputationHelper {
        }
    }

    pub fn compute(&mut self, helper: &ComputationHelper) {

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
    }
}
