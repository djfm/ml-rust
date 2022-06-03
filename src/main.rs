pub mod examples;
pub mod ml;
pub mod util;
pub mod plotter;

pub fn main() {
    examples::mnist::train();
}
