extern crate num;

use num::complex::Complex;

use self::num::{One, Zero};
use crate::{Register, UnitaryBuilder};

/// Apply the QFFT circuit to a given Register using the builder.
pub fn qfft(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
    let rs = b.split_all(r);
    let rs = rec_qfft(b, rs);
    b.merge(rs).unwrap()
}

fn rec_qfft(b: &mut dyn UnitaryBuilder, rs: Vec<Register>) -> Vec<Register> {
    unimplemented!()
}


///// Apply the QFFT circuit to a given Register using the builder.
//pub fn qfft(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
//    let mut rs = b.split_all(r);
//    rs.reverse();
//    let rs = rec_qfft(b, vec![], rs);
//    b.merge(rs).unwrap()
//}
//wrap_and_invert!(pub qfft_op, pub qfft_inv, qfft, r);
//
//fn rec_qfft(
//    builder: &mut dyn UnitaryBuilder,
//    mut used_qs: Vec<Register>,
//    mut remaining_qs: Vec<Register>,
//) -> Vec<Register> {
//    let q = remaining_qs.pop();
//    if let Some(q) = q {
//        let mut q = builder.hadamard(q);
//
//        let mut pushing_qs = vec![];
//        for (i, cq) in remaining_qs.into_iter().enumerate() {
//            let m = (i + 2) as u64;
//            let mut cbuilder = builder.with_condition(cq);
//            // Rm is a 2x2 matrix, so cannot panic on unwrap.
//            q = cbuilder.mat("Rm", q, make_rm_mat(m)).unwrap();
//            pushing_qs.push(cbuilder.release_register());
//        }
//        pushing_qs.reverse();
//        let qs = pushing_qs;
//
//        used_qs.push(q);
//        rec_qfft(builder, used_qs, qs)
//    } else {
//        used_qs
//    }
//}

fn make_rm_mat(m: u64) -> Vec<Complex<f64>> {
    let phi = 2.0 * std::f64::consts::PI / f64::from(1 << m);
    vec![
        Complex::one(),
        Complex::zero(),
        Complex::zero(),
        Complex { re: 0.0, im: phi }.exp(),
    ]
}

#[cfg(test)]
mod qfft_tests {
    use super::*;
    use crate::{run_local, run_local_with_init, CircuitError, OpBuilder, QuantumState};

    #[test]
    fn flat_test() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let (r, h) = b.register_and_handle(10)?;

        let r = qfft(&mut b, r);

        let assert_fn = |(_, v): (usize, Complex<f64>)| {
            assert!((v.norm_sqr() - (1.0 / 1024.0)).abs() < 1e-10);
        };

        let (s, _) = run_local::<f64>(&r)?;
        s.get_state(true)
            .into_iter()
            .enumerate()
            .for_each(assert_fn);

        let (s, _) = run_local_with_init::<f64>(&r, &[h.make_init_from_index(511)?])?;
        s.get_state(true)
            .into_iter()
            .enumerate()
            .for_each(assert_fn);
        Ok(())
    }

    #[test]
    fn cos_test() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let (r, h) = b.register_and_handle(10)?;

        let r = qfft(&mut b, r);

        let (r, m) = b.stochastic_measure(r);

        let assert_fn = |(_, v): (usize, Complex<f64>)| {
            assert!((v.norm_sqr() - (1.0 / 1024.0)).abs() < 1e-10);
        };

        let state = (0..1 << 10).map(|indx| {
            Complex { re: (f64::from(indx)*std::f64::consts::PI/10.0).cos(), im: 0.0 }
        }).collect();
        let (_, mut measurements) = run_local_with_init::<f64>(&r,
                                                               &[h.make_init_from_state(state)?])?;
        let state = measurements.pop_stochastic_measurements(m).unwrap();
        state.into_iter()
            .enumerate()
            .for_each(|(indx, v)| {
                println!("{}\t{}", indx, v)
            });
        assert!(false);
        Ok(())
    }
}
