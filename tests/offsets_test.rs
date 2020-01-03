extern crate num;
extern crate qip;

use qip::pipeline::{run_with_statebuilder, LocalQuantumState, MeasuredResults};
use qip::*;

fn assert_almost_eq(a: f64, b: f64, prec: i32) {
    let mult = 10.0f64.powi(prec);
    let (a, b) = (a * mult, b * mult);
    let (a, b) = (a.round(), b.round());
    assert_eq!(a / mult, b / mult);
}

fn run_with_offsets<P: Precision>(
    n: u64,
    r: &Register,
    input_region: (usize, usize),
    output_region: (usize, usize),
) -> Result<(LocalQuantumState<P>, MeasuredResults<P>), CircuitError> {
    run_with_statebuilder(r, |rs| {
        Ok(
            LocalQuantumState::<P>::new_from_intitial_states_and_regions(
                n,
                &[],
                input_region,
                output_region,
            ),
        )
    })
}

#[test]
fn test_offsets() -> Result<(), CircuitError> {
    let mut b = OpBuilder::new();

    let r = b.qubit();
    let r = b.hadamard(r);
    let r2 = b.qubit();
    let r = b.merge(vec![r, r2])?;
    let n = 2;
    let (s, _) = run_with_offsets::<f64>(n, &r, (0, 1 << n as usize), (0, 1 << n as usize))?;
    let (s1, _) = run_with_offsets::<f64>(n, &r, (0, 1 << n as usize), (0, 1 << (n as usize - 1)))?;
    let (s2, _) = run_with_offsets::<f64>(
        n,
        &r,
        (0, 1 << n as usize),
        (1 << (n as usize - 1), 1 << n as usize),
    )?;

    let m = s.state_magnitude();
    let m1 = s1.state_magnitude();
    let m2 = s2.state_magnitude();

    assert_almost_eq(m, 1.0, 8);
    assert_almost_eq(m1, 0.5, 8);
    assert_almost_eq(m2, 0.5, 8);

    Ok(())
}
