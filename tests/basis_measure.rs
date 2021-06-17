extern crate num;
extern crate qip;

mod utils;

use qip::*;
use utils::assert_almost_eq;

#[test]
fn test_measure_true() -> Result<(), CircuitError> {
    let mut b = OpBuilder::new();
    let q = b.qubit();
    let q = b.hadamard(q);
    let (q, m) = b.measure_basis(q, std::f64::consts::FRAC_PI_4);
    let (_, measured) = run_local(&q)?;

    let (m, p) = measured.get_measurement(&m).unwrap();
    assert_eq!(m, 1);
    assert_almost_eq(p, 1.0, 10);

    Ok(())
}

#[test]
fn test_measure_false() -> Result<(), CircuitError> {
    let mut b = OpBuilder::new();
    let q = b.qubit();
    let q = b.hadamard(q);
    let (q, m) = b.measure_basis(q, -std::f64::consts::FRAC_PI_4);
    let (_, measured) = run_local(&q)?;

    let (m, p) = measured.get_measurement(&m).unwrap();
    assert_eq!(m, 0);
    assert_almost_eq(p, 1.0, 10);

    Ok(())
}
