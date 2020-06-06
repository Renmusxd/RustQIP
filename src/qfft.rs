use num::complex::Complex;
use num::{One, Zero};

use crate::{Register, UnitaryBuilder};

/// Apply the QFFT circuit to a given Register using the builder.
pub fn qfft<B: UnitaryBuilder>(builder: &mut B, r: Register) -> Register {
    let mut rs = builder.split_all(r);
    rs.reverse();
    let rs = rec_qfft(builder, vec![], rs);
    builder.merge(rs).unwrap()
}

fn rec_qfft<B: UnitaryBuilder>(
    builder: &mut B,
    mut used_qs: Vec<Register>,
    mut remaining_qs: Vec<Register>,
) -> Vec<Register> {
    let q = remaining_qs.pop();
    if let Some(q) = q {
        let mut q = builder.hadamard(q);

        let mut pushing_qs = vec![];
        for (i, cq) in remaining_qs.into_iter().enumerate() {
            let m = (i + 2) as u64;
            let mut cbuilder = builder.with_condition(cq);
            // Rm is a 2x2 matrix, so cannot panic on unwrap.
            q = cbuilder.mat("Rm", q, make_rm_mat(m)).unwrap();
            pushing_qs.push(cbuilder.release_register());
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
