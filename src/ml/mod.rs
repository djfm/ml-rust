pub mod ad_number;
pub mod ad;
mod scalar_number;
pub mod math;
pub mod ff_network;
mod layer;

pub use crate::ml::{
    math::{
        NumberLike,
        NumberFactory,
        CellActivation,
        LayerActivation,
        ErrorFunction,
    },
    ad::{
        AD,
    },
    ad_number::{
        ADNumber,
        ADNumberFactory,
    },
    ff_network::{
        Network,
        TrainingConfig,
        TrainingSample,
    },
    layer::{
        Layer,
        LayerConfig,
    },
};

mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x + x;
        println!("{:#?}", ad);
        assert_eq!(y.diff(&x), 2.0);
        assert_eq!((y + ad.create_variable(7.0)).diff(&x), 2.0);
        assert_eq!((y + ad.create_constant(7.0)).diff(&x), 2.0);
        assert_eq!((y + 7.0).diff(&x), 2.0);
        assert_eq!((y + 7.0).scalar(), 11.0);
    }

    #[test]
    fn test_mul_sub() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x - x;
        let z = ad.create_variable(3.0);
        let t = x * z - y;
        println!("{:#?}", ad);
        assert_eq!(t.diff(&x), 3.0);
        assert_eq!(t.diff(&z), 2.0);
        assert_eq!(t.diff(&y), -1.0);
    }

    #[test]
    fn test_dx2_dx() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x * x;
        let dy_dx = y.diff(&x);
        assert_eq!(dy_dx, 4.0);
    }

    #[test]
    fn test_dx2y_dx_dx2y_dy() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = ad.create_variable(3.0);
        let o = x * x * y;
        assert_eq!(o.diff(&x), 12.0);
        assert_eq!(o.diff(&y), 4.0);
    }

    #[test]
    fn test_much_more_complex_diff() {
        let ad = AD::new();
        let x = ad.create_variable(3.0);
        let y = ad.create_variable(4.0);
        let o = y / (x.exp() - y);
        assert_eq!(o.diff(&x), -0.310507656);
        assert_eq!(o.diff(&y), 0.077626914);
    }

    #[test]
    fn test_neg() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let minus_x = -x;
        assert_eq!(minus_x.scalar(), -2.0);
        assert_eq!(minus_x.diff(&x), -1.0);
    }
}
