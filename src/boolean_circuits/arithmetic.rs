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
/// `a,b < M`, `M > 0`.
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
wrap_and_invert!(pub add_mod_op, pub add_mod_inv, (add_mod), ra, rb, rm);

/// Maps `|a>|b>|M>|p>` to `|a>|b>|M>|(p + ba) mod M>`
/// With `a[n+1]`, `b[k]`, `M[n]`, and `p[n+1]`, and `a,p < M`, `M > 0`.
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
            let (ra, rm, rt, rc) = program!(b, ra, rm, rt, rc;
                lshift_op ra;
                control add_inv rt[indx], rc, rm, ra;
                control x ra[n], rt[indx];
                add_op rc, rm, ra;
            )?;
            b.pop_name_scope();
            Ok((ra, rb, rm, rp, rt, rc))
        })?;
        let (ra, rb, rm, rp, rt, rc) = rs;

        b.return_temp_register(rc, false);
        b.return_temp_register(rt, false);
        b.pop_name_scope();
        Ok((ra, rb, rm, rp))
    }
}
wrap_and_invert!(pub times_mod_op, pub times_mod_inv, (times_mod), ra, rb, rm, rp);

/// Right shift the qubits in a register (or left shift by providing a negative number).
pub fn rshift(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
    b.push_name_scope("rshift");
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
    b.pop_name_scope();
    b.merge(rs.into_iter().map(|r| r.unwrap()).collect())
        .unwrap()
}
wrap_and_invert!(pub rshift_op, pub lshift_op, rshift, r);

/// Performs |a>|b> -> |a>|a ^ b>, which for b=0 is a copy operation.
pub fn copy(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
    if ra.n() != rb.n() {
        CircuitError::make_err(format!(
            "Expected ra.n = rb.n, but found {} and {}",
            ra.n(), rb.n()
        ))
    } else {
        b.push_name_scope("copy");
        let ras = b.split_all(ra);
        let rbs = b.split_all(rb);
        let (ras, rbs) = ras.into_iter().zip(rbs.into_iter())
            .fold((vec![], vec![]),
                  |(mut ras, mut rbs), (ra, rb)| {
                      let (ra, rb) = b.cnot(ra, rb);
                      ras.push(ra);
                      rbs.push(rb);
                      (ras, rbs)
                  });
        let ra = b.merge(ras)?;
        let rb = b.merge(rbs)?;
        b.pop_name_scope();
        Ok((ra, rb))
    }
}
wrap_and_invert!(pub copy_op, pub copy_inv, (copy), ra, rb);

/// Performs |a>|m>|s> -> |a>|m>|(s + a*a) % m>.
pub fn square_mod(b: &mut dyn UnitaryBuilder, ra: Register, rm: Register, rs: Register) -> Result<(Register, Register, Register), CircuitError> {
    let n = rm.n();
    if ra.n() != n + 1 {
        CircuitError::make_err(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        ))
    } else if rs.n() != n + 1 {
        CircuitError::make_err(format!(
            "Expected rs.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rs.n()
        ))
    } else {
        b.push_name_scope("square_mod");
        let rt = b.get_temp_register(n, false);
        let (ra, rm, rs, rt) = program!(b, ra, rm, rs, rt;
            copy_op ra[0 .. n], rt;
            times_mod_op ra, rt, rm, rs;
            copy_inv ra[0 .. n], rt;
        )?;
        b.return_temp_register(rt, false);
        b.pop_name_scope();
        Ok((ra, rm, rs))
    }
}
wrap_and_invert!(pub square_mod_op, pub square_mod_inv, (square_mod), ra, rm, rs);

/// Maps |a>|b>|m>|p>|0> -> |a>|b>|m>|p>|(p*a^b) mod m>
pub fn exp_mod(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register, rm: Register, rp: Register, re: Register) -> Result<(Register, Register, Register, Register, Register), CircuitError> {
    let n = rm.n();
    let k = rb.n();
    if ra.n() != n+1 {
        CircuitError::make_err(format!(
            "Expected ra.n = rm.n + 1 = {}, but found {}",
            n + 1,
            ra.n()
        ))
    } else if rp.n() != n+1 {
        CircuitError::make_err(format!(
            "Expected ro.n = rm.n + 1 = {}, but found {}",
            n + 1,
            rp.n()
        ))
    } else if re.n() != n+1 {
        CircuitError::make_err(format!(
            "Expected re.n = rm.n + 1 = {}, but found {}",
            n + 1,
            re.n()
        ))
    } else {
        b.push_name_scope("exp_mod");
        let ret = if k == 1 {
            program!(b, ra, rb, rm, rp, re;
                control(0) copy_op rb[0], rp, re;
                control times_mod_op rb[0], ra, rp, rm, re;
            )
        } else {
            let ru = b.get_temp_register(n + 1, false);
            let rv = b.get_temp_register(n + 1, false);

            let (ra, rb, rm, rp, re, ru, rv) = program!(b, ra, rb, rm, rp, re, ru, rv;
                control(0) copy_op rb[0], rp, rv;
                control times_mod_op rb[0], ra, rp, rm, re;
                square_mod_op ra, rm, ru;
                exp_mod_op ru, rb[1 .. k], rm, rv, re;
                square_mod_inv ra, rm, ru;
                control times_mod_inv rb[0], ra, rp, rm, re;
                control(0) copy_inv rb[0], rp, rv;
            )?;

            b.return_temp_register(ru, false);
            b.return_temp_register(rv, false);

            Ok((ra, rb, rm, rp, re))
        };
        b.pop_name_scope();
        ret
    }
}
wrap_and_invert!(pub exp_mod_op, pub exp_mod_inv, (exp_mod), ra, rb, rm, rp, re);


#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    use crate::pipeline::{InitialState, MeasurementHandle};
    use crate::sparse_state::run_sparse_local_with_init;
    use num::One;
    use crate::utils::extract_bits;

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

    fn assert_on_registers_and_filter<
        F: Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>,
        G: Fn(Vec<u64>, Vec<u64>, u64) -> (),
        FilterFn: Fn(&[u64]) -> bool
    >(
        b: &mut OpBuilder,
        rs: Vec<Register>,
        f: F,
        assertion: G,
        filter: FilterFn
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
            let filter_measurements: Vec<_> = index_groups.iter().map(|indices| {
                extract_bits(indx, indices)
            }).collect();
            if filter(&filter_measurements) {
                let (state, measurements) = run_sparse_local_with_init::<f64>(
                    &r,
                    &[(indices.clone(), InitialState::Index(indx))],
                )?;
                let before_measurements = before_measurements
                    .iter()
                    .map(|m| measurements.get_measurement(m).unwrap().0)
                    .collect();
                assert_eq!(filter_measurements, before_measurements);
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
            }
            Ok(())
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
        assert_on_registers_and_filter(b, rs, f, assertion, |_| true)
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

        assert_on_registers_and_filter(&mut b, vec![rc, ra, rb], add_op, |befores, afters, _| {
            let c = befores[0];
            let a = befores[1];
            let b = befores[2];

            let q_c = afters[0];
            let q_a = afters[1];
            let q_b = afters[2];

            dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
        }, |befores| {
            let c = befores[0];
            let b = befores[2];
            (b & (1 << n)) == 0 && (c & (1 << (n - 1)) == 0)
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

        assert_on_registers_and_filter(&mut b, vec![rc, ra, rb], add_op, |befores, afters, _| {
            let c = befores[0];
            let a = befores[1];
            let b = befores[2];

            let q_c = afters[0];
            let q_a = afters[1];
            let q_b = afters[2];

            dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << (n + 1)));
            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, (a + c + b) % (1 << (n + 1)));
        }, |befores| {
            let c = befores[0];
            let b = befores[2];
            (b & (1 << n)) == 0 && c < 2
        })?;
        Ok(())
    }

    #[test]
    fn test_mod_add() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(1)?;
        let rb = b.register(2)?;
        let rm = b.register(1)?;

        assert_on_registers_and_filter(
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

                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + b) % m);
                assert_eq!(q_m, m);
                assert_eq!(full >> 4, 0);
            }, |befores| {
                let a = befores[0];
                let b = befores[1];
                let m = befores[2];
                a < m && b < m && (b >> 1) == 0
            }
        )?;
        Ok(())
    }

    #[test]
    fn test_mod_add_larger() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        let rm = b.register(2)?;

        assert_on_registers_and_filter(
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

                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + b) % m);
                assert_eq!(q_m, m);
                assert_eq!(full >> 7, 0);
            }, |befores| {
                let a = befores[0];
                let b = befores[1];
                let m = befores[2];
                a < m && b < m && (b >> 2) == 0
            }
        )?;
        Ok(())
    }

    #[test]
    fn test_rshift() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;

        assert_on_registers(&mut b, vec![r], rshift_op, |befores, afters, _| {
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

        assert_on_registers(&mut b, vec![r], lshift_op, |befores, afters, _| {
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
        let ra = b.register(n + 1)?;
        let rb = b.register(k)?;
        let rm = b.register(n)?;
        let rp = b.register(n + 1)?;

        assert_on_registers_and_filter(
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

                assert_eq!(q_a, a);
                assert_eq!(q_b, b);
                assert_eq!(q_m, m);
                assert_eq!(q_p, (p + a * b) % m);
                assert_eq!(full >> (3 * n + k + 2), 0);
            }, |befores| {
                let a = befores[0];
                let m = befores[2];
                let p = befores[3];
                a < m && p < m && m > 0
            }
        )?;
        Ok(())
    }

    #[test]
    fn test_mod_squared() -> Result<(), CircuitError> {
        let n = 2;
        let mut b = OpBuilder::new();
        let ra = b.register(n + 1)?;
        let rm = b.register(n)?;
        let rs = b.register(n + 1)?;

        assert_on_registers_and_filter(
            &mut b,
            vec![ra, rm, rs],
            square_mod_op,
            |befores, afters, full| {
                let a = befores[0];
                let m = befores[1];
                let s = befores[2];

                let q_a = afters[0];
                let q_m = afters[1];
                let q_s = afters[2];

                dbg!(a, m, s, q_a, q_m, q_s, (s + a*a) % m);

                assert_eq!(q_a, a);
                assert_eq!(q_m, m);
                assert_eq!(q_s, (s + a*a) % m);
                assert_eq!(full >> (3 * n + 1), 0);
            }, |befores| {
                let a = befores[0];
                let m = befores[1];
                let s = befores[2];
                m > 0 && a < m && s < m
            }
        )?;
        Ok(())
    }

    fn assert_exp_mod(n: u64, k: u64, befores: &[u64], afters: &[u64], full: u64) {
        let c_a = befores[0];
        let c_b = befores[1];
        let c_m = befores[2];
        let c_p = befores[3];

        let q_a = afters[0];
        let q_b = afters[1];
        let q_m = afters[2];
        let q_p = afters[3];
        let q_e = afters[4];

        let expected = (c_p * c_a.pow(c_b as u32)) % c_m;
        assert_eq!(q_a, c_a);
        assert_eq!(q_b, c_b);
        assert_eq!(q_m, c_m);
        assert_eq!(q_p, c_p);
        assert_eq!(q_e, expected);
        assert_eq!(full >> (4 * n + 3 + k), 0);
    }

    fn filter_exp_mod(befores: &[u64]) -> bool {
        let a = befores[0];
        let m = befores[2];
        let p = befores[3];
        let e = befores[4];
        m > 0 && a < m && p < m && e == 0
    }

    #[test]
    fn test_exp_mod_base() -> Result<(), CircuitError> {
        let n = 1;
        let k = 1;
        let mut b = OpBuilder::new();
        let ra = b.register(n + 1)?;
        let rb = b.register(k)?;
        let rm = b.register(n)?;
        let rp = b.register(n + 1)?;
        let re = b.register(n + 1)?;

        assert_on_registers_and_filter(
            &mut b,
            vec![ra, rb, rm, rp, re],
            exp_mod_op,
            |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
            filter_exp_mod
        )?;
        Ok(())
    }

    #[test]
    fn test_exp_mod_base_larger() -> Result<(), CircuitError> {
        let n = 2;
        let k = 1;
        let mut b = OpBuilder::new();
        let ra = b.register(n + 1)?;
        let rb = b.register(k)?;
        let rm = b.register(n)?;
        let rp = b.register(n + 1)?;
        let re = b.register(n + 1)?;

        assert_on_registers_and_filter(
            &mut b,
            vec![ra, rb, rm, rp, re],
            exp_mod_op,
            |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
            filter_exp_mod
        )?;
        Ok(())
    }

    #[test]
    fn test_exp_small_rec() -> Result<(), CircuitError> {
        let n = 1;
        let k = 2;
        let mut b = OpBuilder::new();
        let ra = b.register(n + 1)?;
        let rb = b.register(k)?;
        let rm = b.register(n)?;
        let rp = b.register(n + 1)?;
        let re = b.register(n + 1)?;

        assert_on_registers_and_filter(
            &mut b,
            vec![ra, rb, rm, rp, re],
            exp_mod_op,
            |befores, afters, full| assert_exp_mod(n, k, &befores, &afters, full),
            filter_exp_mod
        )?;
        Ok(())
    }

    // The n=k=2 case takes too long to test completely.
}
