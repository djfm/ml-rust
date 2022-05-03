mod ad_number;
mod ad;

pub use ad::{
    AD,
};

mod tests {

    #[test]
    fn test_add() {
        use super::{AD};
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
        use super::{AD};
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
}
