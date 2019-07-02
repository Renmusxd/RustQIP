extern crate num;

use num::complex::Complex;

use crate::qubits::{Qubit, UnitaryBuilder};

pub fn qfft<B: UnitaryBuilder>(builder: &mut B, q: Qubit) -> Qubit {
    let mut qs = builder.split_all(q);
    qs.reverse();
    let qs = rec_qfft(builder, vec![], qs);
    builder.merge(qs)
}

fn rec_qfft<B: UnitaryBuilder>(builder: &mut B, mut used_qs: Vec<Qubit>, mut remaining_qs: Vec<Qubit>) -> Vec<Qubit> {
    let q = remaining_qs.pop();
    if let Some(q) = q {
        let mut q = builder.hadamard(q);

        let mut pushing_qs = vec![];
        for (i, cq) in remaining_qs.into_iter().enumerate() {
            let m = (i + 2) as u64;
            let mut cbuilder = builder.with_context(cq);
            // Rm is a 2x2 matrix, so cannot panic on unwrap.
            q = cbuilder.mat("Rm", q, make_rm_mat(m).as_slice()).unwrap();
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

pub fn make_rm_mat(m: u64) -> Vec<Complex<f64>>{
    let phi = 2.0 * std::f64::consts::PI / f64::from(1 << m);
    vec![Complex {
        re: 1.0,
        im: 0.0
    }, Complex {
        re: 0.0,
        im: 0.0
    }, Complex {
        re: 0.0,
        im: 0.0
    }, Complex {
        re: 0.0,
        im: phi
    }.exp()]
}