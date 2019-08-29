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

/// Right shift the qubits in a register (or left shift by providing a negative number).
pub fn rshift(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
    let n = r.n();
    let mut rs: Vec<Option<Register>> = b.split_all(r).into_iter().map(|r| Some(r)).collect();
    let mut hit: Vec<bool> = rs.iter().map(|_| false ).collect();
    (0..n-1).rev().for_each(|indx| {
        let ra = rs[indx as usize].take().unwrap();
        let offset = (indx as i64 - 1) % (n as i64);
        let offset = if offset < 0 { offset + n as i64 } else { offset } as u64;
        let rb = rs[offset as usize].take().unwrap();
        let (ra, rb) = b.swap(ra, rb).unwrap();
        rs[indx as usize] = Some(ra);
        rs[offset as usize] = Some(rb);
    });
    b.merge(rs.into_iter().map(|r| r.unwrap()).collect()).unwrap()
}
wrap_and_invert!(rshift_op, lshift_op, rshift, r);

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    use crate::pipeline::{
        get_opfns_and_frontier, get_required_state_size_from_frontier,
        InitialState,
    };
    use crate::utils::{extract_bits, flip_bits};
    use num::One;

    fn get_mapping_from_indices(r: &Register, indices: &[u64]) -> Result<Vec<u64>, CircuitError> {
        let v = (0..1 << indices.len())
            .into_iter()
            .try_fold(vec![], |mut acc, indx| {
                let mut indices = indices.to_vec();
                indices.reverse();
                let (state, _) =
                    run_local_with_init::<f64>(&r, &[(indices, InitialState::Index(indx as u64))])
                        .unwrap();
                let pos = state
                    .get_state(false)
                    .into_iter()
                    .position(|v| v == Complex::one());
                match pos {
                    Some(pos) => {
                        let (rs, _) = get_opfns_and_frontier(&r);
                        let n = get_required_state_size_from_frontier(&rs);
                        acc.push(flip_bits(n as usize, pos as u64));
                        Ok(acc)
                    }
                    None => CircuitError::make_err(format!("Error any mapping for {}", indx)),
                }
            })?;
        Ok(v)
    }

    fn get_mapping<P: Precision>(r: &Register) -> Result<Vec<u64>, CircuitError> {
        let indices: Vec<_> = (0..r.n()).collect();
        get_mapping_from_indices(r, &indices)
    }

    #[test]
    fn test_carry_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.qubit();
        let rcp = b.qubit();

        let (rc, ra, rb, rcp) = carry(&mut b, rc, ra, rb, rcp)?;

        let r = b.merge(vec![rc, ra, rb, rcp])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:04b}\t{:04b}", indx, mapping);
            let indx = indx as u64;
            let c = 0 != indx & 1;
            let a = 0 != indx & (1 << 1);
            let b = 0 != indx & (1 << 2);
            let cp = 0 != indx & (1 << 3);

            let q_c = 0 != mapping & 1;
            let q_a = 0 != mapping & (1 << 1);
            let q_b = 0 != mapping & (1 << 2);
            let q_cp = 0 != mapping & (1 << 3);

            let c_func = |a: bool, b: bool, c: bool| -> bool { (a & b) ^ (c & (a ^ b)) };
            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, b);
            assert_eq!(q_cp, cp ^ c_func(a, b, c));
        });
        Ok(())
    }

    #[test]
    fn test_inv_carry_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.qubit();
        let rcp = b.qubit();

        let (rc, ra, rb, rcp) = program!(&mut b, rc, ra, rb, rcp;
            carry_op rc, ra, rb, rcp;
            carry_inv rc, ra, rb, rcp;
        )?;

        let r = b.merge(vec![rc, ra, rb, rcp])?;
        run_debug(&r)?;
        let inv_mapping = get_mapping::<f64>(&r)?;

        inv_mapping
            .into_iter()
            .enumerate()
            .for_each(|(indx, result)| {
                assert_eq!(indx as u64, result);
            });

        Ok(())
    }

    #[test]
    fn test_sum_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.qubit();

        let (rc, ra, rb) = sum(&mut b, rc, ra, rb);

        let r = b.merge(vec![rc, ra, rb])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:04b}\t{:04b}", indx, mapping);
            let indx = indx as u64;
            let c = 0 != indx & 1;
            let a = 0 != indx & (1 << 1);
            let b = 0 != indx & (1 << 2);

            let q_c = 0 != mapping & 1;
            let q_a = 0 != mapping & (1 << 1);
            let q_b = 0 != mapping & (1 << 2);

            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, a ^ b ^ c);
        });
        Ok(())
    }

    #[test]
    fn test_add_1m() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.register(2)?;

        let (rc, ra, rb) = add(&mut b, rc, ra, rb)?;

        let r = b.merge(vec![rc, ra, rb])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:04b}\t{:04b}", indx, mapping);
            let indx = indx as u64;
            let c = 0 != indx & 1;
            let a = 0 != indx & (1 << 1);
            let b = extract_bits(indx, &[2, 3]);

            let q_c = 0 != mapping & 1;
            let q_a = 0 != mapping & (1 << 1);
            let q_b = extract_bits(mapping, &[2, 3]);

            let num = |x: bool| {
                if x {
                    1
                } else {
                    0
                }
            };

            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, (b + num(c) + num(a)) % 4)
        });
        Ok(())
    }

    #[test]
    fn test_add_2m() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.register(2)?;
        let ra = b.register(2)?;
        let rb = b.register(3)?;

        let (rc, ra, rb) = add(&mut b, rc, ra, rb)?;

        let r = b.merge(vec![rc, ra, rb])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:07b}\t{:07b}", indx, mapping);
            let indx = indx as u64;
            let c = extract_bits(indx, &[0, 1]);
            let a = extract_bits(indx, &[2, 3]);
            let b = extract_bits(indx, &[4, 5, 6]);

            if (b & (1 << 3)) == 0 && (c & (1 << 1) == 0) {
                let q_c = extract_bits(mapping, &[0, 1]);
                let q_a = extract_bits(mapping, &[2, 3]);
                let q_b = extract_bits(mapping, &[4, 5, 6]);

                assert_eq!(q_c, c);
                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + c + b) % 8);
            }
        });
        Ok(())
    }

    #[test]
    fn test_mod_add() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(1)?;
        let rb = b.register(2)?;
        let rm = b.register(1)?;

        let (ra, rb, rm) = add_mod(&mut b, ra, rb, rm)?;

        let r = b.merge(vec![ra, rb, rm])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;
        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:06b}\t{:06b}", indx, mapping);
            let indx = indx as u64;
            let a = extract_bits(indx, &[0]);
            let b = extract_bits(indx, &[1, 2]);
            let m = extract_bits(indx, &[3]);

            if b < m {
                let q_a = extract_bits(mapping, &[0]);
                let q_b = extract_bits(mapping, &[1, 2]);
                let q_m = extract_bits(mapping, &[3]);

                dbg!(a, b, m, q_a, q_b, q_m, (a + b) % m);

                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + b) % m);
                assert_eq!(q_m, m);
            }
        });
        Ok(())
    }

    #[test]
    fn test_mod_add_larger() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        let rm = b.register(2)?;

        let (ra, rb, rm) = add_mod(&mut b, ra, rb, rm)?;

        let r = b.merge(vec![ra, rb, rm])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;
        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:010b}\t{:010b}", indx, mapping);
            let indx = indx as u64;
            let a = extract_bits(indx, &[0, 1]);
            let b = extract_bits(indx, &[2, 3, 4]);
            let m = extract_bits(indx, &[5, 6]);

            if a < m && b < m {
                let q_a = extract_bits(mapping, &[0, 1]);
                let q_b = extract_bits(mapping, &[2, 3, 4]);
                let q_m = extract_bits(mapping, &[5, 6]);

                dbg!(a, b, m, q_a, q_b, q_m, (a + b) % m);

                assert_eq!(q_a, a);
                assert_eq!(q_b, (a + b) % m);
                assert_eq!(q_m, m);
            }
        });
        Ok(())
    }

    #[test]
    fn test_rshift_simple() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;
        let r = rshift(&mut b, r);

        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:05b}\t{:05b}", indx, mapping);
            let indx = indx as u64;
            let expected_output = indx << 1;
            let expected_output = (expected_output | (expected_output >> n)) & ((1 << n) - 1);

            assert_eq!(mapping, expected_output);
        });
        Ok(())
    }

    #[test]
    fn test_rshift_wrapped() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;
        let r = program!(&mut b, r;
            rshift_op r;
        )?;

        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:05b}\t{:05b}", indx, mapping);
            let indx = indx as u64;
            let expected_output = indx << 1;
            let expected_output = (expected_output | (expected_output >> n)) & ((1 << n) - 1);

            assert_eq!(mapping, expected_output);
        });
        Ok(())
    }

    #[test]
    fn test_lshift_wrapped() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let n = 5;
        let r = b.register(n)?;
        let r = program!(&mut b, r;
            lshift_op r;
        )?;

        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:05b}\t{:05b}", indx, mapping);
            let indx = indx as u64;
            let expected_output = indx >> 1;
            let expected_output = expected_output | ((indx & 1) << (n-1));

            assert_eq!(mapping, expected_output);
        });
        Ok(())
    }
}
