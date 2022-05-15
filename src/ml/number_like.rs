pub trait NumberLike: Copy + Clone {
    fn scalar(&self) -> f32;
}
