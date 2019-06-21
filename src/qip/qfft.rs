extern crate num;

use num::complex::Complex;

use crate::qubits;
use crate::qubits::{Qubit, UnitaryBuilder};
use crate::state_ops::from_reals;


pub fn qfft<B: UnitaryBuilder>(builder: &mut B, q: Qubit) -> Qubit {
    let mut qs = builder.split_all(q);
    qs.reverse();
    let qs = rec_qfft(builder, vec![], qs);
    let mut qs: Vec<Option<Qubit>> = qs.into_iter().map(|q| Some(q)).collect();

    for i in 0 .. qs.len() >> 1 {
        let inv_i = qs.len() - i - 1;
        let qa = qs[i].take();
        let qb = qs[inv_i].take();
        if let Some((qa, qb)) = qa.and_then(|a| qb.map(|b| (a,b))) {
            let (qa, qb) = builder.swap(qa, qb);
            qs[i] = Some(qa);
            qs[inv_i] = Some(qb);
        }
    }
    let qs = qs.into_iter().map(|q| q.unwrap()).collect();

    builder.merge(qs)
}

fn rec_qfft<B: UnitaryBuilder>(builder: &mut B, mut used_qs: Vec<Qubit>, mut remaining_qs: Vec<Qubit>) -> Vec<Qubit> {
    let q = remaining_qs.pop();
    if let Some(q) = q {
        let offset = used_qs.len();
        let q = builder.hadamard(q);
        let mut cbuilder = builder.with_context(q);
        let rn = remaining_qs.len();
        let qs: Vec<Qubit> = remaining_qs.into_iter().enumerate().map(|(i, q)| {
            let m = i + 2;
            cbuilder.mat(q, make_rm_mat(m as u64).as_slice())
        }).collect();

        let q = cbuilder.release_qubit();
        used_qs.push(q);
        rec_qfft(builder, used_qs, qs)
    } else {
        used_qs
    }
}

pub fn make_rm_mat(m: u64) -> Vec<Complex<f64>>{
    let phi = -2.0 * std::f64::consts::PI / 2.0f64.powi(m as i32);
    vec![Complex::<f64> {
        re: 1.0,
        im: 0.0
    }, Complex::<f64> {
        re: 0.0,
        im: 0.0
    }, Complex::<f64> {
        re: 0.0,
        im: 0.0
    }, Complex::<f64> {
        re: 0.0,
        im: phi
    }.exp()]
}