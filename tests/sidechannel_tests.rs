extern crate qip;
use qip::pipeline::MeasurementHandle;
use qip::qubits::QubitHandle;
use qip::*;

fn assert_almost_eq(a: f64, b: f64, prec: i32) {
    let mult = 10.0f64.powi(prec);
    let (a, b) = (a * mult, b * mult);
    let (a, b) = (a.round(), b.round());
    assert_eq!(a / mult, b / mult);
}

fn setup_cswap_sidechannel_circuit(
    vec_n: u64,
) -> Result<(Qubit, QubitHandle, QubitHandle, MeasurementHandle), InvalidValueError> {
    // Setup inputs
    let mut b = OpBuilder::new();
    let q1 = b.qubit(1)?;
    let q2 = b.qubit(vec_n)?;
    let q3 = b.qubit(vec_n)?;

    // We will want to feed in some inputs later.
    let h2 = q2.handle();
    let h3 = q3.handle();

    // Define circuit
    let q1 = b.hadamard(q1);

    // Make a qubit whose sole use is for sidechannels
    let q4 = b.qubit(1)?;
    let q4 = b.hadamard(q4);
    let (q4, h4) = b.measure(q4);

    let mut c = b.with_condition(q1);
    let qs = c.classical_sidechannel(
        vec![q2, q3],
        &[h4],
        Box::new(|b, mut qs, _ms| {
            let q3 = qs.pop().unwrap();
            let q2 = qs.pop().unwrap();
            let (q2, q3) = b.swap(q2, q3)?;
            Ok(vec![q2, q3])
        }),
    );
    let q1 = c.release_qubit();

    let q1 = b.hadamard(q1);

    let (q1, m1) = b.measure(q1);

    Ok((q1, h2, h3, m1))
}

#[test]
fn test_cswap_sidechannel() -> Result<(), InvalidValueError> {
    let vec_n = 3;

    let (q1, h2, h3, m1) = setup_cswap_sidechannel_circuit(vec_n)?;

    // Run circuit
    let (_, measured) = run_local_with_init::<f64>(
        &q1,
        &[h2.make_init_from_index(0)?, h3.make_init_from_index(0)?],
    )?;

    let (m, p) = measured.get_measurement(&m1).unwrap();
    assert_eq!(m, 0);
    assert_almost_eq(p, 1.0, 10);
    Ok(())
}

#[test]
fn test_cswap_sidechannel_unaligned() -> Result<(), InvalidValueError> {
    let vec_n = 3;

    let (q1, h2, h3, m1) = setup_cswap_sidechannel_circuit(vec_n)?;

    // Run circuit
    let (_, measured) = run_local_with_init::<f64>(
        &q1,
        &[h2.make_init_from_index(0)?, h3.make_init_from_index(1)?],
    )?;

    let (m, p) = measured.get_measurement(&m1).unwrap();
    assert!(m == 0 || m == 1);
    assert_almost_eq(p, 0.5, 10);
    Ok(())
}
