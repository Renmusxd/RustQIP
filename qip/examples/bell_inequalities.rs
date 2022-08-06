#[cfg(feature = "macros")]
use qip::prelude::*;
#[cfg(feature = "macros")]
use qip::prelude::{CircuitBuilder, CircuitError};
#[cfg(feature = "macros")]
use qip_macros::*;
#[cfg(feature = "macros")]
use qip::macros::program_ops::*;
use std::num::NonZeroUsize;

#[cfg(feature = "macros")]
fn circuit1() -> Result<Vec<f64>, CircuitError>{
    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(2).unwrap();

    let q = b.qubit();
    let ra = b.register(n);
    let r = program!(&mut b; r;
        not r;
        h r[0];
        control not r[0], r[1];
        rz(std::f64::consts::FRAC_PI_3) r[1];
        h r;
    )?;
    let (r, m_handle) = b.measure_stochastic(r);

    //run and get probabilities
    let (_, mut measured) = run_local::<f64>(&r).unwrap();
    measured.pop_stochastic_measurements(m_handle).unwrap()
}

#[cfg(feature = "macros")]
fn circuit2() -> Result<Vec<f64>, CircuitError>{
    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(2).unwrap();

    let r = b.register(n);
    let r = program!( &mut b; r;
        not r;
        h r[0];
        control not r[0], r[1];
        rz(2. * std::f64::consts::FRAC_PI_3) r[1];
        h r;
    )?;
    let (r, m_handle) = b.measure_stochastic(r);

    // run and get probabilities
    let (_, mut measured) = run_local::<f64>(&r).unwrap();
    measured.pop_stochastic_measurements(m_handle).unwrap()
}
// #[cfg(feature = "macros")]
fn circuit3() -> Result<Vec<f64>, CircuitError> {
    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(2).unwrap();
    let r = b.register(n);

    let r = program!(&mut b; r;
        not r;
        h r[0];
        control not r[0], r[1];
        rz(std::f64::consts::FRAC_PI_3) r[0];
        rz(2. * std::f64::consts::FRAC_PI_3) r[1];
        h r;
    ).unwrap();
    let (r, m_handle) = b.measure_stochastic(r); //returns (Self::Register, Self::StochasticMeasurementHandle)
    let (_, measurements) = b.calculate_state();
    let (m, _) = measurements.get_stochastic_measurement(m_handle);
}

#[cfg(not(feature = "macros"))]
fn main() -> () {}

/// Bell inequality:
/// |P(a, b) - P(a, c)| - P(b, c) <= 1
/// To get P(a, b), P(a, c) and P(b, c) we use the 3 circuits presented in:
/// https://arxiv.org/pdf/1712.05642.pdf - II.IMPLEMENTED EXPERIMENTS -> 3. Bell's inequality
///
/// A violation of the inequality is expected in any quantum computer or simulator.
#[cfg(feature = "macros")]
fn main() -> Result<(), CircuitError> {
    println!("Bell inequality: |P(a, b) - P(a, c)| - P(b, c) <= 1");

    let a_b = circuit1();
    let p_of_a_b = (a_b?[0] + a_b?[3]) - (a_b?[1] + a_b?[2]);
    println!("P(a, b) = {:.2}", p_of_a_b);

    let a_c = circuit2();
    let p_of_a_c = (a_c?[0] + a_c?[3]) - (a_c?[1] + a_c?[2]);
    println!("P(a, c) = {:.2}", p_of_a_c);

    let b_c = circuit3();
    let p_of_b_c = (b_c?[0] + b_c?[3]) - (b_c?[1] + b_c?[2]);
    println!("P(b, c) = {:.2}", p_of_b_c);

    let left_side = (p_of_a_b - p_of_a_c).abs() - p_of_b_c;
    println!(
        "|{:.2} - {:.2}| - ({:.2}) = {:.2} IS NOT <= 1",
        p_of_a_b, p_of_a_c, p_of_b_c, left_side
    );

    assert!(left_side > 1.0);

    Ok(())
}
