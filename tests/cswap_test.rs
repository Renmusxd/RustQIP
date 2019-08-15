extern crate num;
extern crate qip;

use qip::pipeline::MeasurementHandle;
use qip::qubits::RegisterHandle;
use qip::*;

fn assert_almost_eq(a: f64, b: f64, prec: i32) {
    let mult = 10.0f64.powi(prec);
    let (a, b) = (a * mult, b * mult);
    let (a, b) = (a.round(), b.round());
    assert_eq!(a / mult, b / mult);
}

fn setup_cswap_circuit(
    vec_n: u64,
) -> Result<(Register, RegisterHandle, RegisterHandle, MeasurementHandle), CircuitError> {
    // Setup inputs
    let mut b = OpBuilder::new();
    let q1 = b.register(1)?;
    let q2 = b.register(vec_n)?;
    let q3 = b.register(vec_n)?;

    // We will want to feed in some inputs later.
    let h2 = q2.handle();
    let h3 = q3.handle();

    // Define circuit
    let q1 = b.hadamard(q1);

    let (q1, _) = condition(&mut b, q1, (q2, q3), |c, (q2, q3)| c.swap(q2, q3)).unwrap();
    let q1 = b.hadamard(q1);

    let (q1, m1) = b.measure(q1);

    Ok((q1, h2, h3, m1))
}

#[test]
fn test_cswap_aligned() -> Result<(), CircuitError> {
    // Setup inputs
    let (q1, h2, h3, m1) = setup_cswap_circuit(3)?;

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
fn test_cswap_orthogonal() -> Result<(), CircuitError> {
    // Setup inputs
    let (q1, h2, h3, m1) = setup_cswap_circuit(3)?;

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

/// sin((2 * pi * i * w / L) + d) normalized
fn sin_wave(n: u64, w: f64, d: f64) -> Vec<Complex<f64>> {
    let l = 1 << n;
    let state: Vec<_> = (0..l)
        .map(|i| -> Complex<f64> {
            let v = (d + (std::f64::consts::PI * w * i as f64 / l as f64)).sin();
            Complex { re: v, im: 0.0 }
        })
        .collect();
    let mag: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    let mag = mag.sqrt();
    state.iter().map(|c| -> Complex<f64> { c / mag }).collect()
}

#[test]
fn test_cswap_waves() -> Result<(), CircuitError> {
    // Setup inputs
    let (q1, h2, h3, m1) = setup_cswap_circuit(3)?;

    let s1 = sin_wave(3, 1.0, 0.0);
    let s2 = sin_wave(3, 2.0, 0.0);

    // Run circuit
    let (_, measured) = run_local_with_init::<f64>(
        &q1,
        &[h2.make_init_from_state(s1)?, h3.make_init_from_state(s2)?],
    )?;

    let (m, p) = measured.get_measurement(&m1).unwrap();
    assert!(m == 0 || m == 1);
    assert_almost_eq(p, 0.5, 3);

    Ok(())
}
