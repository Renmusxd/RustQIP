extern crate num;

use num::complex::Complex;

use crate::qubits::{Qubit, UnitaryBuilder};
use crate::types::Precision;

pub fn qfft<P: Precision, B: UnitaryBuilder<P>>(builder: &mut B, q: Qubit<P>) -> Qubit<P> {
    let mut qs = builder.split_all(q);
    qs.reverse();
    let qs = rec_qfft(builder, vec![], qs);
    builder.merge(qs)
}

fn rec_qfft<P: Precision, B: UnitaryBuilder<P>>(builder: &mut B, mut used_qs: Vec<Qubit<P>>, mut remaining_qs: Vec<Qubit<P>>) -> Vec<Qubit<P>> {
    let q = remaining_qs.pop();
    if let Some(q) = q {
        let mut q = builder.hadamard(q);

        let mut pushing_qs = vec![];
        for (i, cq) in remaining_qs.into_iter().enumerate() {
            let m = (i + 2) as u64;
            let mut cbuilder = builder.with_context(cq);
            // Rm is a 2x2 matrix, so cannot panic on unwrap.
            q = cbuilder.mat(q, make_rm_mat(m).as_slice()).unwrap();
            pushing_qs.push(cbuilder.release_qubit());
        }
        pushing_qs.reverse();
        let qs = pushing_qs;

        used_qs.push(q);
        rec_qfft(builder, used_qs, qs)
    } else {
        used_qs
    }
}

pub fn make_rm_mat<P: Precision>(m: u64) -> Vec<Complex<P>>{
    let phi = P::from(2.0 * std::f64::consts::PI / (1 << m) as f64).unwrap();
    vec![Complex::<P> {
        re: P::one(),
        im: P::zero()
    }, Complex::<P> {
        re: P::zero(),
        im: P::zero()
    }, Complex::<P> {
        re: P::zero(),
        im: P::zero()
    }, Complex::<P> {
        re: P::zero(),
        im: phi
    }.exp()]
}