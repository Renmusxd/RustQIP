/// A collection of circuits from chapter 6.4 of "Quantum Computing: A gentle introduction"
/// by Eleanor Rieffle and Wolfgang Polak.
use crate::macros::common_ops::x;
use crate::*;

/// Add together ra and rb using rc as carry, result is in rb.
/// This works when the highest order bit of rb and rc are both |0>. Undefined behavior otherwise.
/// ra and rc have m qubits, rb has m+1 qubits.
pub fn add(
    b: &mut dyn UnitaryBuilder,
    rc: Register,
    ra: Register,
    rb: Register,
) -> Result<(Register, Register, Register), CircuitError> {
    b.push_name_scope("add");
    let result = match (rc.n(), ra.n(), rb.n()) {
        (1, 1, 2) => {
            let (rc, ra, rb) = program!(b, rc, ra, rb;
                carry_op rc, ra, rb[0], rb[1];
                sum_op rc, ra, rb[0];
            )?;
            Ok((rc, ra, rb))
        }
        (nc, na, nb) if nc == na && nc + 1 == nb => {
            let n = nc;
            let (rc, ra, rb) = program!(b, rc, ra, rb;
                carry_op rc[0], ra[0], rb[0], rc[1];
                add_op rc[1..n], ra[1..n], rb[1..=n];
                carry_inv rc[0], ra[0], rb[0], rc[1];
                sum_op rc[0], ra[0], rb[0];
            )?;
            Ok((rc, ra, rb))
        }
        (nc, na, nb) => CircuitError::make_err(format!(
            "Expected rc[n] ra[n] and rb[n+1], but got ({},{},{})",
            nc, na, nb
        )),
    };
    b.pop_name_scope();
    result
}
wrap_and_invert!(pub add_op, pub add_inv, (add), ra, rb, rc);

fn sum(
    b: &mut dyn UnitaryBuilder,
    rc: Register,
    ra: Register,
    rb: Register,
) -> (Register, Register, Register) {
    b.push_name_scope("sum");
    let (ra, rb) = b.cx(ra, rb);
    let (rc, rb) = b.cx(rc, rb);
    b.pop_name_scope();
    (rc, ra, rb)
}
wrap_fn!(sum_op, sum, rc, ra, rb);

fn carry(
    b: &mut dyn UnitaryBuilder,
    rc: Register,
    ra: Register,
    rb: Register,
    rcp: Register,
) -> Result<(Register, Register, Register, Register), CircuitError> {
    b.push_name_scope("carry");
    let (rc, ra, rb, rcp) = program!(b, rc, ra, rb, rcp;
        control x |ra, rb,| rcp;
        control x ra, rb;
        control x |rc, rb,| rcp;
        control x ra, rb;
    )?;
    b.pop_name_scope();
    Ok((rc, ra, rb, rcp))
}
wrap_and_invert!(carry_op, carry_inv, (carry), rc, ra, rb, rcp);

/// Addition of ra and rb modulo rm. Conditions are:
/// 0 <= a
/// a,b < M
pub fn add_mod(
    b: &mut dyn UnitaryBuilder,
    ra: Register,
    rb: Register,
    rm: Register,
) -> Result<(Register, Register, Register), CircuitError> {
    if ra.n() != rm.n() {
        CircuitError::make_err(format!(
            "Expected rm.n == ra.n == {}, found rm.n={}.",
            ra.n(),
            rm.n()
        ))
    } else if rb.n() != ra.n() + 1 {
        CircuitError::make_err(format!(
            "Expected rb.n == ra.n + 1== {}, found rm.n={}.",
            ra.n() + 1,
            rb.n()
        ))
    } else {
        b.push_name_scope("add_mod");
        let n = ra.n();

        let rt = b.get_temp_register(1, false);
        let rc = b.get_temp_register(n, false);

        let (ra, rb, rm, rt, rc) = program!(b, ra, rb, rm, rt, rc;
            add_op rc, ra, rb;
            add_inv rc, rm, rb;
            control x rb[n], rt;
            control add_op rt, rc, rm, rb;
            add_inv rc, ra, rb;
            control(0) x rb[n], rt;
            add_op rc, ra, rb;
        )?;
        b.return_temp_register(rt, false);
        b.return_temp_register(rc, false);
        b.pop_name_scope();
        Ok((ra, rb, rm))
    }
}
wrap_fn!(pub add_mod_op, (add_mod), ra, rb, rm);

/// Maps `|a>|b>|M>|p>` to `|a>|b>|M>|(p + ba) mod M>`
/// With `a[n+1]`, `b[k]`, `M[n]`, and `p[n+1]`, and `a < M`.
pub fn times_mod(
    b: &mut dyn UnitaryBuilder,
    ra: Register,
    rb: Register,
    rm: Register,
    rp: Register,
) -> Result<(Register, Register, Register, Register), CircuitError> {
    let n = rm.n();
    let k = rb.n();
    if ra.n() != n + 1 {
        CircuitError::make_err(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        ))
    } else if rp.n() != n + 1 {
        CircuitError::make_err(format!(
            "Expected rp.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rp.n()
        ))
    } else {
        b.push_name_scope("times_mod");
        let rt = b.get_temp_register(k, false);
        let rc = b.get_temp_register(n, false);

        let rs = (ra, rb, rm, rp, rt, rc);
        let rs = (0..k).try_fold(rs, |rs, indx| {
            let (ra, rb, rm, rp, rt, rc) = rs;
            b.push_name_scope(&format!("first_{}", indx));
            let ret = program!(b, ra, rb, rm, rp, rt, rc;
                add_inv rc, rm, ra;
                control x ra[n], rt[indx];
                control add_op rt[indx], rc, rm, ra;
                control add_mod_op rb[indx], ra[0 .. n], rp, rm;
                rshift_op ra;
            );
            b.pop_name_scope();
            ret
        })?;
        let rs = (0..k).rev().try_fold(rs, |rs, indx| {
            let (ra, rb, rm, rp, rt, rc) = rs;
            b.push_name_scope(&format!("second_{}", indx));
            let ret = program!(b, ra, rb, rm, rp, rt, rc;
                lshift_op ra;
                control add_inv rt[indx], rc, rm, ra;
                control x ra[n], rt[indx];
                add_op rc, rm, ra;
            );
            b.pop_name_scope();
            ret
        })?;
        let (ra, rb, rm, rp, rt, rc) = rs;

        b.return_temp_register(rc, false);
        b.return_temp_register(rt, false);
        b.pop_name_scope();
        Ok((ra, rb, rm, rp))
    }
}
wrap_fn!(pub times_mod_op, (times_mod), ra, rb, rm, rp);

/// Right shift the qubits in a register (or left shift by providing a negative number).
pub fn rshift(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
    let n = r.n();
    let mut rs: Vec<Option<Register>> = b.split_all(r).into_iter().map(Some).collect();
    (0..n - 1).rev().for_each(|indx| {
        let ra = rs[indx as usize].take().unwrap();
        let offset = (indx as i64 - 1) % (n as i64);
        let offset = if offset < 0 {
            offset + n as i64
        } else {
            offset
        } as u64;
        let rb = rs[offset as usize].take().unwrap();
        let (ra, rb) = b.swap(ra, rb).unwrap();
        rs[indx as usize] = Some(ra);
        rs[offset as usize] = Some(rb);
    });
    b.merge(rs.into_iter().map(|r| r.unwrap()).collect())
        .unwrap()
}
wrap_and_invert!(pub rshift_op, pub lshift_op, rshift, r);

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    use crate::pipeline::{InitialState, MeasurementHandle};
    use crate::sparse_state::run_sparse_local_with_init;
    use num::One;

    fn measure_each(
        b: &mut OpBuilder,
        rs: Vec<Register>,
    ) -> (Vec<Register>, Vec<MeasurementHandle>) {
        rs.into_iter()
            .fold((vec![], vec![]), |(mut rs, mut ms), r| {
                let (r, m) = b.measure(r);
                rs.push(r);
                ms.push(m);
                (rs, ms)
            })
    }

    fn assert_on_registers<
        F: Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>,
        G: Fn(Vec<u64>, Vec<u64>, u64) -> (),
    >(
        b: &mut OpBuilder,
        rs: Vec<Register>,
        f: F,
        assertion: G,
    ) -> Result<(), CircuitError> {
        let n: u64 = rs.iter().map(|r| r.n()).sum();
        let index_groups: Vec<_> = rs.iter().map(|r| r.indices.clone()).collect();
        let (rs, before_measurements) = measure_each(b, rs);
        let rs = f(b, rs)?;
        let (rs, after_measurements) = measure_each(b, rs);
        let r = b.merge(rs)?;
        run_debug(&r)?;
        let indices: Vec<_> = (0..n).collect();
        (0..1 << n).into_iter().try_for_each(|indx| {
            let (state, measurements) = run_sparse_local_with_init::<f64>(
                &r,
                &[(indices.clone(), InitialState::Index(indx))],
            )?;
            let before_measurements = before_measurements
                .iter()
                .map(|m| measurements.get_measurement(m).unwrap().0)
                .collect();
            let after_measurements = after_measurements
                .iter()
                .map(|m| measurements.get_measurement(m).unwrap().0)
                .collect();
            let state_index = state
                .get_state(true)
                .into_iter()
                .position(|v| v == Complex::one())
                .unwrap() as u64;
            assertion(before_measurements, after_measurements, state_index);
            Ok(())
        })
    }

    #[test]
    fn test_carry_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.qubit();
        let rcp = b.qubit();

        assert_on_registers(
            &mut b,
            vec![rc, ra, rb, rcp],
            carry_op,
            |befores, afters, _| {
                let c = 0 != befores[0];
                let a = 0 != befores[1];
                let b = 0 != befores[2];
                let cp = 0 != befores[3];

                let q_c = 0 != afters[0];
                let q_a = 0 != afters[1];
                let q_b = 0 != afters[2];
                let q_cp = 0 != afters[3];

                let c_func = |a: bool, b: bool, c: bool| -> bool { (a & b) ^ (c & (a ^ b)) };
                dbg!(a, b, c, cp, q_a, q_b, q_c, q_cp, cp ^ c_func(a, b, c));
                assert_eq!(q_c, c);
                assert_eq!(q_a, a);
                assert_eq!(q_b, b);
                assert_eq!(q_cp, cp ^ c_func(a, b, c));
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_sum_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.qubit();

        assert_on_registers(&mut b, vec![rc, ra, rb], sum_op, |befores, afters, _| {
            let c = 0 != befores[0];
            let a = 0 != befores[1];
            let b = 0 != befores[2];

            let q_c = 0 != afters[0];
            let q_a = 0 != afters[1];
            let q_b = 0 != afters[2];

            dbg!(c, a, b, q_c, q_a, q_b, a ^ b ^ c);
            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, a ^ b ^ c);
        })?;
        Ok(())
    }

    #[test]
    fn test_add_1m() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.register(2)?;

        assert_on_registers(&mut b, vec![rc, ra, rb], add_op, |befores, afters, _| {
            let c = 0 != befores[0];
            let a = 0 != befores[1];
            let b = befores[2];

            let q_c = 0 != afters[0];
            let q_a = 0 != afters[1];
            let q_b = afters[2];

            let num = |x: bool| {
                if x {
                    1
                } else {
                    0
                }
            };

            dbg!(c, a, b, q_c, q_a, q_b, (b + num(c) + num(a)) % 4);
            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, (b + num(c) + num(a)) % 4)
        })?;
        Ok(())
    }

    #[test]
    fn test_add_2m() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 2;
        let rc = b.register(n)?;
        let ra = b.register(n)?;
        let rb = b.register(n + 1)?;

        assert_on_registers(&mut b, vec![rc, ra, rb], add_op, |befores, afters, _| {
            let c = befores[0];
            let a = befores[1];
            let b = befores[2];

            let q_c = afters[0];
            let q_a = afters[1];
            let q_b = afters[2];

            dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
            if (b & (1 << n)) == 0 && (c & (1 << (n - 1)) == 0) {
                assert_eq!(q_c, c);
                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
            } else {
                println!("Skipped");
            }
        })?;
        Ok(())
    }

    #[test]
    fn test_add_3m() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 3;
        let rc = b.register(n)?;
        let ra = b.register(n)?;
        let rb = b.register(n + 1)?;

        assert_on_registers(&mut b, vec![rc, ra, rb], add_op, |befores, afters, full| {
            let c = befores[0];
            let a = befores[1];
            let b = befores[2];

            let q_c = afters[0];
            let q_a = afters[1];
            let q_b = afters[2];

            println!("Full: {:010b}", full);
            dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
            if (b & (1 << n)) == 0 && c < 2 {
                assert_eq!(q_c, c);
                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
            } else {
                println!("Skipped");
            }
        })?;
        Ok(())
    }

    #[test]
    fn test_mod_add() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(1)?;
        let rb = b.register(2)?;
        let rm = b.register(1)?;

        assert_on_registers(
            &mut b,
            vec![ra, rb, rm],
            add_mod_op,
            |befores, afters, full| {
                let a = befores[0];
                let b = befores[1];
                let m = befores[2];

                let q_a = afters[0];
                let q_b = afters[1];
                let q_m = afters[2];

                println!("Full: {:06b}", full);
                dbg!(a, b, m, q_a, q_b, q_m);
                if a < m && b < m && (b >> 1) == 0 {
                    dbg!((a + b) % m);
                    assert_eq!(q_a, a);
                    assert_eq!(q_b, (a + b) % m);
                    assert_eq!(q_m, m);
                    assert_eq!(full >> 4, 0);
                } else {
                    println!("Skipped");
                }
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_mod_add_larger() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        let rm = b.register(2)?;

        assert_on_registers(
            &mut b,
            vec![ra, rb, rm],
            add_mod_op,
            |befores, afters, full| {
                let a = befores[0];
                let b = befores[1];
                let m = befores[2];

                let q_a = afters[0];
                let q_b = afters[1];
                let q_m = afters[2];

                println!("Full: {:010b}", full);
                dbg!(a, b, m, q_a, q_b, q_m);
                if a < m && b < m && (b >> 2) == 0 {
                    dbg!((a + b) % m);
                    assert_eq!(q_a, a);
                    assert_eq!(q_b, (a + b) % m);
                    assert_eq!(q_m, m);
                    assert_eq!(full >> 7, 0);
                } else {
                    println!("Skipped");
                }
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_rshift() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;

        assert_on_registers(&mut b, vec![r], rshift_op, |befores, afters, full| {
            let expected_output = befores[0] << 1;
            let expected_output = (expected_output | (expected_output >> n)) & ((1 << n) - 1);
            assert_eq!(afters[0], expected_output);
        })?;
        Ok(())
    }

    #[test]
    fn test_lshift() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;

        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;

        assert_on_registers(&mut b, vec![r], lshift_op, |befores, afters, full| {
            let expected_output = befores[0] >> 1;
            let expected_output = expected_output | ((befores[0] & 1) << (n - 1));
            assert_eq!(afters[0], expected_output);
        })?;
        Ok(())
    }

    #[test]
    fn test_mod_times() -> Result<(), CircuitError> {
        let n = 2;
        let k = 2;
        let mut b = OpBuilder::new();
        let (ra, ha) = b.register_and_handle(n + 1)?;
        let (rb, hb) = b.register_and_handle(k)?;
        let (rm, hm) = b.register_and_handle(n)?;
        let (rp, hp) = b.register_and_handle(n + 1)?;

        assert_on_registers(
            &mut b,
            vec![ra, rb, rm, rp],
            times_mod_op,
            |befores, afters, full| {
                let a = befores[0];
                let b = befores[1];
                let m = befores[2];
                let p = befores[3];

                let q_a = afters[0];
                let q_b = afters[1];
                let q_m = afters[2];
                let q_p = afters[3];

                println!("Full: {:017b}", full);
                if a < m && m > 0 {
                    dbg!(a, b, m, p, q_a, q_b, q_m, q_p, (p + a * b) % m);
                    assert_eq!(q_a, a);
                    assert_eq!(q_b, b);
                    assert_eq!(q_m, m);
                    assert_eq!(q_p, p + (p + a * b) % m);
                    assert_eq!(full >> (3 * n + k + 2), 0);
                } else {
                    println!("Skipped");
                }
            },
        )?;
        Ok(())
    }
}
