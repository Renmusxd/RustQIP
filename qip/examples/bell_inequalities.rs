use qip::prelude::*;
#[cfg(feature = "macros")]
use qip_macros::*;
use std::num::NonZeroUsize;

#[cfg(feature = "macros")]
fn circuit1() -> Vec<f64> {
    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(2).unwrap();

    let q = b.qubit();
    let ra = b.register(n);
    let r = program!(&mut b, r;
        not r;
        h r[0];
        control not r[0], r[1];
        rz(std::f64::consts::FRAC_PI_3) r[1];
        h r;
    )
    .unwrap();
    let (r, m_handle) = b.stochastic_measure(r);

    //run and get probabilities
    let (_, mut measured) = run_local::<f64>(&r).unwrap();
    measured.pop_stochastic_measurements(m_handle).unwrap()
}



#[cfg(not(feature = "macros"))]
fn main() {}

