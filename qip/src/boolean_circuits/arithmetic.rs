//! A collection of circuits from chapter 6.4 of "Quantum Computing: A gentle introduction"
//! by Eleanor Rieffle and Wolfgang Polak.

extern crate self as qip;

use crate::errors::CircuitError;
use crate::inverter::RecursiveCircuitBuilder;
use crate::macros::program_ops::*;
use crate::prelude::*;
use qip_macros::*;
use std::num::NonZeroUsize;

macro_rules! register_tuple {
    ($($T:ident),*) => {
        (
            $(<$T as CircuitBuilder>::Register,)*
        )
    }
}

type R2<CB> = register_tuple!(CB, CB);
type R3<CB> = register_tuple!(CB, CB, CB);
type R4<CB> = register_tuple!(CB, CB, CB, CB);
type R5<CB> = register_tuple!(CB, CB, CB, CB, CB);

/// Add together ra and rb using rc as carry, result is in rb.
/// This works when the highest order bit of rb and rc are both |0>. Undefined behavior otherwise.
/// ra and rc have m qubits, rb has m+1 qubits.
#[invert]
pub fn add<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    rc: CB::Register,
    ra: CB::Register,
    rb: CB::Register,
) -> CircuitResult<R3<CB>> {
    match (rc.n(), ra.n(), rb.n()) {
        (1, 1, 2) => {
            let (rc, ra, rb) = program!(&mut *b; rc, ra, rb;
                carry rc, ra, rb[0], rb[1];
                sum rc, ra, rb[0];
            )?;
            Ok((rc, ra, rb))
        }
        (nc, na, nb) if nc == na && nc + 1 == nb => {
            let n = nc;
            let (rc, ra, rb) = program!(&mut *b; rc, ra, rb;
                carry rc[0], ra[0], rb[0], rc[1];
                add rc[1..n], ra[1..n], rb[1..=n];
                carry_inv rc[0], ra[0], rb[0], rc[1];
                sum rc[0], ra[0], rb[0];
            )?;
            Ok((rc, ra, rb))
        }
        (nc, na, nb) => Err(CircuitError::new(format!(
            "Expected rc[n] ra[n] and rb[n+1], but got ({},{},{})",
            nc, na, nb
        ))),
    }
}

fn sum<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    rc: CB::Register,
    ra: CB::Register,
    rb: CB::Register,
) -> CircuitResult<R3<CB>> {
    program!(&mut *b; rc, ra, rb;
        control x ra, rb;
        control x rc, rb;
    )
}

#[invert]
fn carry<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    rc: CB::Register,
    ra: CB::Register,
    rb: CB::Register,
    rcp: CB::Register,
) -> CircuitResult<R4<CB>> {
    let (rc, ra, rb, rcp) = program!(&mut *b; rc, ra, rb, rcp;
        control x [ra, rb] rcp;
        control x ra, rb;
        control x [rc, rb] rcp;
        control x ra, rb;
    )?;

    Ok((rc, ra, rb, rcp))
}

/// Addition of ra and rb modulo rm. Conditions are:
/// `a,b < M`, `M > 0`.
#[invert]
pub fn add_mod<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    ra: CB::Register,
    rb: CB::Register,
    rm: CB::Register,
) -> CircuitResult<R3<CB>> {
    if ra.n() != rm.n() {
        Err(CircuitError::new(format!(
            "Expected rm.n == ra.n == {}, found rm.n={}.",
            ra.n(),
            rm.n()
        )))
    } else if rb.n() != ra.n() + 1 {
        Err(CircuitError::new(format!(
            "Expected rb.n == ra.n + 1== {}, found rm.n={}.",
            ra.n() + 1,
            rb.n()
        )))
    } else {
        let n = ra.n();

        let rt = b.make_zeroed_temp_qubit();
        let rc = b.make_zeroed_temp_register(NonZeroUsize::new(n).unwrap());

        let (ra, rb, rm, rt, rc) = program!(&mut *b; ra, rb, rm, rt, rc;
            add rc, ra, rb;
            add_inv rc, rm, rb;
            control x rb[n], rt;
            control add rt, rc, rm, rb;
            add_inv rc, ra, rb;
            control(0) x rb[n], rt;
            add rc, ra, rb;
        )?;
        b.return_zeroed_temp_register(rt);
        b.return_zeroed_temp_register(rc);

        Ok((ra, rb, rm))
    }
}

/// Maps `|a>|b>|M>|p>` to `|a>|b>|M>|(p + ba) mod M>`
/// With `a[n+1]`, `b[k]`, `M[n]`, and `p[n+1]`, and `a,p < M`, `M > 0`.
#[invert]
pub fn times_mod<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    ra: CB::Register,
    rb: CB::Register,
    rm: CB::Register,
    rp: CB::Register,
) -> CircuitResult<R4<CB>> {
    let n = rm.n();
    let k = rb.n();
    if ra.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        )))
    } else if rp.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected rp.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rp.n()
        )))
    } else {
        let rt = b.make_zeroed_temp_register(NonZeroUsize::new(k).unwrap());
        let rc = b.make_zeroed_temp_register(NonZeroUsize::new(n).unwrap());

        let rs = (ra, rb, rm, rp, rt, rc);
        let rs = (0..k).try_fold(rs, |rs, indx| {
            let (ra, rb, rm, rp, rt, rc) = rs;

            program!(&mut *b; ra, rb, rm, rp, rt, rc;
                add_inv rc, rm, ra;
                control x ra[n], rt[indx];
                control add rt[indx], rc, rm, ra;
                control add_mod rb[indx], ra[0 .. n], rp, rm;
                rshift ra;
            )
        })?;
        let rs = (0..k).rev().try_fold(rs, |rs, indx| {
            let (ra, rb, rm, rp, rt, rc) = rs;

            let (ra, rm, rt, rc) = program!(&mut *b; ra, rm, rt, rc;
                lshift ra;
                control add_inv rt[indx], rc, rm, ra;
                control x ra[n], rt[indx];
                add rc, rm, ra;
            )?;

            Ok((ra, rb, rm, rp, rt, rc))
        })?;
        let (ra, rb, rm, rp, rt, rc) = rs;

        b.return_zeroed_temp_register(rc);
        b.return_zeroed_temp_register(rt);

        Ok((ra, rb, rm, rp))
    }
}

/// Right shift the qubits in a register (or left shift by providing a negative number).
#[invert(lshift)]
pub fn rshift<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    r: CB::Register,
) -> CircuitResult<CB::Register> {
    let n = r.n();
    let mut rs: Vec<Option<CB::Register>> = b.split_all_register(r).into_iter().map(Some).collect();
    (0..n - 1).rev().for_each(|indx| {
        let ra = rs[indx].take().unwrap();
        let offset = (indx as i64 - 1) % (n as i64);
        let offset = if offset < 0 {
            offset + n as i64
        } else {
            offset
        } as u64;
        let rb = rs[offset as usize].take().unwrap();
        let (ra, rb) = b.swap(ra, rb).unwrap();
        rs[indx] = Some(ra);
        rs[offset as usize] = Some(rb);
    });

    Ok(b.merge_registers(rs.into_iter().flatten()).unwrap())
}

/// Performs |a>|b> -> |a>|a ^ b>, which for b=0 is a copy operation.
#[invert]
pub fn copy<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    ra: CB::Register,
    rb: CB::Register,
) -> CircuitResult<R2<CB>> {
    if ra.n() != rb.n() {
        Err(CircuitError::new(format!(
            "Expected ra.n = rb.n, but found {} and {}",
            ra.n(),
            rb.n()
        )))
    } else {
        let ras = b.split_all_register(ra);
        let rbs = b.split_all_register(rb);
        let (ras, rbs) = ras.into_iter().zip(rbs.into_iter()).try_fold(
            (vec![], vec![]),
            |(mut ras, mut rbs), (ra, rb)| {
                let (ra, rb) = b.cnot(ra, rb)?;
                ras.push(ra);
                rbs.push(rb);
                Ok((ras, rbs))
            },
        )?;
        let ra = b.merge_registers(ras).unwrap();
        let rb = b.merge_registers(rbs).unwrap();

        Ok((ra, rb))
    }
}

/// Performs |a>|m>|s> -> |a>|m>|(s + a*a) % m>.
#[invert]
pub fn square_mod<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    ra: CB::Register,
    rm: CB::Register,
    rs: CB::Register,
) -> CircuitResult<R3<CB>> {
    let n = rm.n();
    if ra.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        )))
    } else if rs.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected rs.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rs.n()
        )))
    } else {
        let rt = b.make_zeroed_temp_register(NonZeroUsize::new(n).unwrap());
        let (ra, rm, rs, rt) = program!(&mut *b; ra, rm, rs, rt;
            copy ra[0 .. n], rt;
            times_mod ra, rt, rm, rs;
            copy_inv ra[0 .. n], rt;
        )?;
        b.return_zeroed_temp_register(rt);

        Ok((ra, rm, rs))
    }
}

/// Maps |a>|b>|m>|p>|0> -> |a>|b>|m>|p>|(p*a^b) mod m>
#[invert]
pub fn exp_mod<P: Precision, CB: RecursiveCircuitBuilder<P>>(
    b: &mut CB,
    ra: CB::Register,
    rb: CB::Register,
    rm: CB::Register,
    rp: CB::Register,
    re: CB::Register,
) -> CircuitResult<R5<CB>> {
    let n = rm.n();
    let k = rb.n();
    if ra.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        )))
    } else if rp.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected ro.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rp.n()
        )))
    } else if re.n() != n + 1 {
        Err(CircuitError::new(format!(
            "Expected re.n = rm.n + 1 = {}, but found {}",
            n + 1,
            re.n()
        )))
    } else if k == 1 {
        program!(&mut *b; ra, rb, rm, rp, re;
            control(0) copy rb[0], rp, re;
            control times_mod rb[0], ra, rp, rm, re;
        )
    } else {
        let ru = b.make_zeroed_temp_register(NonZeroUsize::new(n + 1).unwrap());
        let rv = b.make_zeroed_temp_register(NonZeroUsize::new(n + 1).unwrap());

        let (ra, rb, rm, rp, re, ru, rv) = program!(&mut *b; ra, rb, rm, rp, re, ru, rv;
            control(0) copy rb[0], rp, rv;
            control times_mod rb[0], ra, rp, rm, re;
            square_mod ra, rm, ru;
            exp_mod ru, rb[1 .. k], rm, rv, re;
            square_mod_inv ra, rm, ru;
            control times_mod_inv rb[0], ra, rp, rm, re;
            control(0) copy_inv rb[0], rp, rv;
        )?;

        b.return_zeroed_temp_register(ru);
        b.return_zeroed_temp_register(rv);

        Ok((ra, rb, rm, rp, re))
    }
}

#[cfg(test)]
mod arithmetic_tests {
    // use super::*;
    // use crate::builder::Qudit;
    // use crate::utils::{extract_bits, flip_bits};
    // use num_traits::One;
    // fn measure_each<CB>(
    //     b: &mut CB,
    //     rs: Vec<CB::Register>,
    // ) -> (Vec<CB::Register>, Vec<CB::MeasurementHandle>)
    //     where
    //         CB: MeasurementBuilder,
    // {
    //     rs.into_iter()
    //         .fold((vec![], vec![]), |(mut rs, mut ms), r| {
    //             let (r, m) = b.measure(r);
    //             rs.push(r);
    //             ms.push(m);
    //             (rs, ms)
    //         })
    // }
    //
    // fn assert_on_registers_and_filter<F, G, FilterFn>(
    //     b: &mut LocalBuilder<f64>,
    //     rs: Vec<Qudit>,
    //     f: F,
    //     assertion: G,
    //     filter: FilterFn,
    // ) -> CircuitResult<()>
    //     where
    //         F: Fn(&mut LocalBuilder<f64>, Vec<Qudit>) -> CircuitResult<Vec<Qudit>>,
    //         G: Fn(Vec<usize>, Vec<usize>, usize),
    //         FilterFn: Fn(&Vec<usize>) -> bool,
    // {
    //     let n: usize = rs.iter().map(|r| r.n()).sum();
    //     let index_groups = rs.iter().map(|r| r.indices().to_vec()).collect::<Vec<_>>();
    //     let (rs, before_measurements) = measure_each(b, rs);
    //     let rs = f(b, rs)?;
    //     let (rs, after_measurements) = measure_each(b, rs);
    //     let r = b
    //         .merge_registers(rs)
    //         .ok_or_else(|| CircuitError::new("No qudits found"))?;
    //     // let indices: Vec<_> = (0..n).collect();
    //     (0..1 << n).into_iter().try_for_each(|indx| {
    //         let filter_measurements: Vec<_> = index_groups
    //             .iter()
    //             .map(|indices| extract_bits(indx, indices))
    //             .collect();
    //         if filter(&filter_measurements) {
    //             let (state, measurements) = b.calculate_state_with_init([(&r, indx)]);
    //             let before_measurements = before_measurements
    //                 .iter()
    //                 .map(|m| measurements.get_measurement(m.clone()).0)
    //                 .collect::<Vec<_>>();
    //             assert_eq!(filter_measurements, before_measurements);
    //             let after_measurements = after_measurements
    //                 .iter()
    //                 .map(|m| measurements.get_measurement(m.clone()).0)
    //                 .collect::<Vec<_>>();
    //             let state_index = state
    //                 .into_iter()
    //                 .position(|v| (v - Complex::one()).norm() < 1e-10)
    //                 .unwrap();
    //             let state_index = flip_bits(n, state_index);
    //             assertion(before_measurements, after_measurements, state_index);
    //         }
    //         Ok(())
    //     })
    // }
    //
    // fn assert_on_registers<F, G>(
    //     b: &mut LocalBuilder<f64>,
    //     rs: Vec<Qudit>,
    //     f: F,
    //     assertion: G,
    // ) -> CircuitResult<()>
    //     where
    //         F: Fn(&mut LocalBuilder<f64>, Vec<Qudit>) -> CircuitResult<Vec<Qudit>>,
    //         G: Fn(Vec<usize>, Vec<usize>, usize),
    // {
    //     assert_on_registers_and_filter(b, rs, f, assertion, |_| true)
    // }
    //
    // #[test]
    // fn test_carry_simple() -> CircuitResult<()> {
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.qubit();
    //     let ra = b.qubit();
    //     let rb = b.qubit();
    //     let rcp = b.qubit();
    //
    //     assert_on_registers(
    //         &mut b,
    //         vec![rc, ra, rb, rcp],
    //         carry,
    //         |befores, afters, _| {
    //             let c = 0 != befores[0];
    //             let a = 0 != befores[1];
    //             let b = 0 != befores[2];
    //             let cp = 0 != befores[3];
    //
    //             let q_c = 0 != afters[0];
    //             let q_a = 0 != afters[1];
    //             let q_b = 0 != afters[2];
    //             let q_cp = 0 != afters[3];
    //
    //             let c_func = |a: bool, b: bool, c: bool| -> bool { (a & b) ^ (c & (a ^ b)) };
    //             dbg!(a, b, c, cp, q_a, q_b, q_c, q_cp, cp ^ c_func(a, b, c));
    //             assert_eq!(q_c, c);
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, b);
    //             assert_eq!(q_cp, cp ^ c_func(a, b, c));
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_sum_simple() -> CircuitResult<()> {
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.qubit();
    //     let ra = b.qubit();
    //     let rb = b.qubit();
    //
    //     assert_on_registers(&mut b, vec![rc, ra, rb], sum, |befores, afters, _| {
    //         let c = 0 != befores[0];
    //         let a = 0 != befores[1];
    //         let b = 0 != befores[2];
    //
    //         let q_c = 0 != afters[0];
    //         let q_a = 0 != afters[1];
    //         let q_b = 0 != afters[2];
    //
    //         dbg!(c, a, b, q_c, q_a, q_b, a ^ b ^ c);
    //         assert_eq!(q_c, c);
    //         assert_eq!(q_a, a);
    //         assert_eq!(q_b, a ^ b ^ c);
    //     })?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_add_1m() -> CircuitResult<()> {
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.qubit();
    //     let ra = b.qubit();
    //     let rb = b.register(NonZeroUsize::new(2).unwrap());
    //
    //     let r = b
    //         .merge_registers([rc, ra, rb])
    //         .ok_or_else(|| CircuitError::new("No qubits"))?;
    //     let mat = make_circuit_matrix(&mut b, &r, |(state, _)| state);
    //     mat.into_iter().enumerate().for_each(|(i, row)| {
    //         println!(
    //             "{:4b}\t{:4b}",
    //             i,
    //             row.into_iter().position(|c| c.norm() > 0.0).unwrap()
    //         );
    //     });
    //     assert!(false);
    //
    //     assert_on_registers(&mut b, vec![rc, ra, rb], add_op, |befores, afters, _| {
    //         let c = befores[0];
    //         let a = befores[1];
    //         let b = befores[2];
    //
    //         let q_c = afters[0];
    //         let q_a = afters[1];
    //         let q_b = afters[2];
    //
    //         dbg!(c, a, b, q_c, q_a, q_b, (b + c + a) % 4);
    //         assert_eq!(q_c, c);
    //         assert_eq!(q_a, a);
    //         assert_eq!(q_b, (b + c + a) % 4)
    //     })?;
    //     Ok(())
    // }

    // #[test]
    // fn test_add_2m() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let n = 2;
    //     let rc = b.register(n)?;
    //     let ra = b.register(n)?;
    //     let rb = b.register(n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![rc, ra, rb],
    //         add_op,
    //         |befores, afters, _| {
    //             let c = befores[0];
    //             let a = befores[1];
    //             let b = befores[2];
    //
    //             let q_c = afters[0];
    //             let q_a = afters[1];
    //             let q_b = afters[2];
    //
    //             dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
    //             assert_eq!(q_c, c);
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
    //         },
    //         |befores| {
    //             let c = befores[0];
    //             let b = befores[2];
    //             (b & (1 << n)) == 0 && (c & (1 << (n - 1)) == 0)
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_add_3m() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let n = 3;
    //     let rc = b.register(n)?;
    //     let ra = b.register(n)?;
    //     let rb = b.register(n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![rc, ra, rb],
    //         add_op,
    //         |befores, afters, _| {
    //             let c = befores[0];
    //             let a = befores[1];
    //             let b = befores[2];
    //
    //             let q_c = afters[0];
    //             let q_a = afters[1];
    //             let q_b = afters[2];
    //
    //             dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
    //             assert_eq!(q_c, c);
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
    //         },
    //         |befores| {
    //             let c = befores[0];
    //             let b = befores[2];
    //             (b & (1 << n)) == 0 && c < 2
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_mod_add() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(1)?;
    //     let rb = b.register(2)?;
    //     let rm = b.register(1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm],
    //         add_mod_op,
    //         |befores, afters, full| {
    //             let a = befores[0];
    //             let b = befores[1];
    //             let m = befores[2];
    //
    //             let q_a = afters[0];
    //             let q_b = afters[1];
    //             let q_m = afters[2];
    //
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, (a + b) % m);
    //             assert_eq!(q_m, m);
    //             assert_eq!(full >> 4, 0);
    //         },
    //         |befores| {
    //             let a = befores[0];
    //             let b = befores[1];
    //             let m = befores[2];
    //             a < m && b < m && (b >> 1) == 0
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_mod_add_larger() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(2)?;
    //     let rb = b.register(3)?;
    //     let rm = b.register(2)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm],
    //         add_mod_op,
    //         |befores, afters, full| {
    //             let a = befores[0];
    //             let b = befores[1];
    //             let m = befores[2];
    //
    //             let q_a = afters[0];
    //             let q_b = afters[1];
    //             let q_m = afters[2];
    //
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, (a + b) % m);
    //             assert_eq!(q_m, m);
    //             assert_eq!(full >> 7, 0);
    //         },
    //         |befores| {
    //             let a = befores[0];
    //             let b = befores[1];
    //             let m = befores[2];
    //             a < m && b < m && (b >> 2) == 0
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_rshift() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let n = 5;
    //     let r = b.register(n)?;
    //
    //     assert_on_registers(&mut b, vec![r], rshift_op, |befores, afters, _| {
    //         let expected_output = befores[0] << 1;
    //         let expected_output = (expected_output | (expected_output >> n)) & ((1 << n) - 1);
    //         assert_eq!(afters[0], expected_output);
    //     })?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_lshift() -> CircuitResult<()> {
    //     let mut b = OpBuilder::new();
    //     let n = 5;
    //     let r = b.register(n)?;
    //
    //     assert_on_registers(&mut b, vec![r], lshift_op, |befores, afters, _| {
    //         let expected_output = befores[0] >> 1;
    //         let expected_output = expected_output | ((befores[0] & 1) << (n - 1));
    //         assert_eq!(afters[0], expected_output);
    //     })?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_mod_times() -> CircuitResult<()> {
    //     let circuit_n = 2;
    //     let mod_k = 2;
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(circuit_n + 1)?;
    //     let rb = b.register(mod_k)?;
    //     let rm = b.register(circuit_n)?;
    //     let rp = b.register(circuit_n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm, rp],
    //         times_mod_op,
    //         |befores, afters, full| {
    //             let a = befores[0];
    //             let b = befores[1];
    //             let m = befores[2];
    //             let p = befores[3];
    //
    //             let q_a = afters[0];
    //             let q_b = afters[1];
    //             let q_m = afters[2];
    //             let q_p = afters[3];
    //
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_b, b);
    //             assert_eq!(q_m, m);
    //             assert_eq!(q_p, (p + a * b) % m);
    //             assert_eq!(full >> (3 * circuit_n + mod_k + 2), 0);
    //         },
    //         |befores| {
    //             let a = befores[0];
    //             let m = befores[2];
    //             let p = befores[3];
    //             a < m && p < m && m > 0
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_mod_squared() -> CircuitResult<()> {
    //     let circuit_n = 2;
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(circuit_n + 1)?;
    //     let rm = b.register(circuit_n)?;
    //     let rs = b.register(circuit_n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rm, rs],
    //         square_mod_op,
    //         |befores, afters, full| {
    //             let a = befores[0];
    //             let m = befores[1];
    //             let s = befores[2];
    //
    //             let q_a = afters[0];
    //             let q_m = afters[1];
    //             let q_s = afters[2];
    //
    //             dbg!(a, m, s, q_a, q_m, q_s, (s + a * a) % m);
    //
    //             assert_eq!(q_a, a);
    //             assert_eq!(q_m, m);
    //             assert_eq!(q_s, (s + a * a) % m);
    //             assert_eq!(full >> (3 * circuit_n + 1), 0);
    //         },
    //         |befores| {
    //             let a = befores[0];
    //             let m = befores[1];
    //             let s = befores[2];
    //             m > 0 && a < m && s < m
    //         },
    //     )?;
    //     Ok(())
    // }
    //
    // fn assert_exp_mod(n: u64, k: u64, befores: &[u64], afters: &[u64], full: u64) {
    //     let c_a = befores[0];
    //     let c_b = befores[1];
    //     let c_m = befores[2];
    //     let c_p = befores[3];
    //
    //     let q_a = afters[0];
    //     let q_b = afters[1];
    //     let q_m = afters[2];
    //     let q_p = afters[3];
    //     let q_e = afters[4];
    //
    //     let expected = (c_p * c_a.pow(c_b as u32)) % c_m;
    //     assert_eq!(q_a, c_a);
    //     assert_eq!(q_b, c_b);
    //     assert_eq!(q_m, c_m);
    //     assert_eq!(q_p, c_p);
    //     assert_eq!(q_e, expected);
    //     assert_eq!(full >> (4 * n + 3 + k), 0);
    // }
    //
    // fn filter_exp_mod(befores: &[u64]) -> bool {
    //     let a = befores[0];
    //     let m = befores[2];
    //     let p = befores[3];
    //     let e = befores[4];
    //     m > 0 && a < m && p < m && e == 0
    // }
    //
    // #[test]
    // fn test_exp_mod_base() -> CircuitResult<()> {
    //     let n = 1;
    //     let k = 1;
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(n + 1)?;
    //     let rb = b.register(k)?;
    //     let rm = b.register(n)?;
    //     let rp = b.register(n + 1)?;
    //     let re = b.register(n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm, rp, re],
    //         exp_mod_op,
    //         |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
    //         filter_exp_mod,
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_exp_mod_base_larger() -> CircuitResult<()> {
    //     let n = 2;
    //     let k = 1;
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(n + 1)?;
    //     let rb = b.register(k)?;
    //     let rm = b.register(n)?;
    //     let rp = b.register(n + 1)?;
    //     let re = b.register(n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm, rp, re],
    //         exp_mod_op,
    //         |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
    //         filter_exp_mod,
    //     )?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_exp_small_rec() -> CircuitResult<()> {
    //     let n = 1;
    //     let k = 2;
    //     let mut b = OpBuilder::new();
    //     let ra = b.register(n + 1)?;
    //     let rb = b.register(k)?;
    //     let rm = b.register(n)?;
    //     let rp = b.register(n + 1)?;
    //     let re = b.register(n + 1)?;
    //
    //     assert_on_registers_and_filter(
    //         &mut b,
    //         vec![ra, rb, rm, rp, re],
    //         exp_mod_op,
    //         |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
    //         filter_exp_mod,
    //     )?;
    //     Ok(())
    // }

    // The n=k=2 case takes too long to test completely.
}
