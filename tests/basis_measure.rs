extern crate num;
extern crate qip;

use qip::*;

fn assert_almost_eq(a: f64, b: f64, prec: i32) {
    let mult = 10.0f64.powi(prec);
    let (a, b) = (a * mult, b * mult);
    let (a, b) = (a.round(), b.round());
    assert_eq!(a / mult, b / mult);
}

#[test]
fn test_measure_true() -> Result<(), &'static str> {
    let mut b = OpBuilder::new();
    let q = b.qubit(1)?;
    let q = b.hadamard(q);
    let (q, m) = b.measure_basis(q, std::f64::consts::FRAC_PI_4);
    let (_, measured) = run_local(&q)?;

    let (m, p) = measured.get_measurement(&m).unwrap();
    assert_eq!(m, 1);
    assert_almost_eq(p, 1.0, 10);

    Ok(())
}

#[test]
fn test_measure_false() -> Result<(), &'static str> {
    let mut b = OpBuilder::new();
    let q = b.qubit(1)?;
    let q = b.hadamard(q);
    let (q, m) = b.measure_basis(q, -std::f64::consts::FRAC_PI_4);
    let (_, measured) = run_local(&q)?;

    let (m, p) = measured.get_measurement(&m).unwrap();
    assert_eq!(m, 0);
    assert_almost_eq(p, 1.0, 10);

    Ok(())
}
