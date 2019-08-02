extern crate num;

use num::complex::Complex;

use self::num::{One, Zero};
use crate::qubits::{Qubit, UnitaryBuilder};

/// Apply the QFFT circuit to a given qubit using the builder.
pub fn qfft<B: UnitaryBuilder>(builder: &mut B, q: Qubit) -> Qubit {
    let mut qs = builder.split_all(q);
    qs.reverse();
    let qs = rec_qfft(builder, vec![], qs);
    builder.merge(qs)
}

fn rec_qfft<B: UnitaryBuilder>(
    builder: &mut B,
    mut used_qs: Vec<Qubit>,
    mut remaining_qs: Vec<Qubit>,
) -> Vec<Qubit> {
    let q = remaining_qs.pop();
    if let Some(q) = q {
        let mut q = builder.hadamard(q);

        let mut pushing_qs = vec![];
        for (i, cq) in remaining_qs.into_iter().enumerate() {
            let m = (i + 2) as u64;
            let mut cbuilder = builder.with_condition(cq);
            // Rm is a 2x2 matrix, so cannot panic on unwrap.
            q = cbuilder.mat("Rm", q, make_rm_mat(m)).unwrap();
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

fn make_rm_mat(m: u64) -> Vec<Complex<f64>> {
    let phi = 2.0 * std::f64::consts::PI / f64::from(1 << m);
    vec![
        Complex::one(),
        Complex::zero(),
        Complex::zero(),
        Complex { re: 0.0, im: phi }.exp(),
    ]
}
