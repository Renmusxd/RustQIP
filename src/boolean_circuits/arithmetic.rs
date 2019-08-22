use crate::*;
use crate::macros::common_ops::x;


/// Add together ra and rb using rc as carry, result is in rb.
pub fn add(b: &mut dyn UnitaryBuilder, rc: Register, ra: Register, rb: Register) -> Result<(Register, Register, Register), CircuitError> {
    match (rc.n(), ra.n(), rb.n()) {
        (1, 1, 2) => {
            let (rc, ra, rb) = program!(b, rc, ra, rb;
                carry_op rc, ra, rb[0], rb[1];
                sum_op rc, ra, rb[0];
            )?;
            Ok((rc, ra, rb))
        },
        (nc, na, nb) if nc == na && nc + 1 == nb => {
            let n = nc;
            let (rc, ra, rb) = program!(b, rc, ra, rb;
                carry_op rc[0], ra[0], rb[0], rc[1];
                add_op rc[1..=n-1], ra[1..=n-1], rb[1..=n];
                inv_carry_op rc[0], ra[0], rb[0], rc[1];
                sum_op rc[0], ra[0], rb[0];
            )?;
            Ok((rc, ra, rb))
        },
        (nc, na, nb) => CircuitError::make_err(format!("Expected rc[n] ra[n] and rb[n+1], but got ({},{},{})", nc, na, nb))
    }
}
wrap_fn!(add_op, (add), ra, rb, rc);

fn sum(b: &mut dyn UnitaryBuilder,
       rc: Register,
       ra: Register,
       rb: Register) -> (Register, Register, Register) {
    let (ra, rb) = b.cx(ra, rb);
    let (rc, rb) = b.cx(rc, rb);
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
    dbg!(&rc, &ra, &rb, &rcp);

    let (rc, ra, rb, rcp) = program!(b, rc, ra, rb, rcp;
        control x |ra, rb,| rcp;
        control x ra, rb;
        control x |rc, rb,| rcp;
        control x ra, rb;
    )?;
    dbg!(&rc, &ra, &rb, &rcp);

    Ok((rc, ra, rb, rcp))
}
wrap_fn!(carry_op, (carry), rc, ra, rb, rcp);


fn inv_carry(
    b: &mut dyn UnitaryBuilder,
    rc: Register,
    ra: Register,
    rb: Register,
    rcp: Register,
) -> Result<(Register, Register, Register, Register), CircuitError> {
    dbg!(&rc, &ra, &rb, &rcp);
    let (rc, ra, rb, rcp) = program!(b, rc, ra, rb, rcp;
        control x ra, rb;
        control x |rc, rb,| rcp;
        control x ra, rb;
        control x |ra, rb,| rcp;
    )?;

    Ok((rc, ra, rb, rcp))
}
wrap_fn!(inv_carry_op, (inv_carry), rc, ra, rb, rcp);


#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    use crate::pipeline::{make_circuit_matrix, InitialState};
    use num::{Zero, One};
    use crate::utils::extract_bits;

    fn get_mapping<P: Precision>(r: &Register) -> Result<Vec<u64>, CircuitError> {
        let indices: Vec<u64> = (0 .. r.n()).collect();
        let v = (0 .. 1 << r.n()).into_iter().try_fold(vec![], |mut acc, indx| {
            let (state, _) =
                run_local_with_init::<f64>(&r, &[(indices.clone(), InitialState::Index(indx))]).unwrap();
            let pos = state.get_state(false).into_iter().position(|v| v == Complex::one());
            match pos {
                Some(pos) => {
                    acc.push(pos as u64);
                    Ok(acc)
                },
                None => {
                    CircuitError::make_err(format!("Error any mapping for {}", indx))
                }
            }
        })?;
        Ok(v)
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
            let c = 0 != indx & (1 << 3);
            let a = 0 != indx & (1 << 2);
            let b = 0 != indx & (1 << 1);
            let cp = 0 != indx & 1;

            let q_c = 0 != mapping & (1 << 3);
            let q_a = 0 != mapping & (1 << 2);
            let q_b = 0 != mapping & (1 << 1);
            let q_cp = 0 != mapping & 1;

            let c_func = |a: bool, b: bool, c: bool| -> bool {
                (a & b) ^ (c & (a ^ b))
            };
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

        let (rc, ra, rb, rcp) = carry(&mut b, rc, ra, rb, rcp)?;
        let (rc, ra, rb, rcp) = inv_carry(&mut b, rc, ra, rb, rcp)?;
        let r = b.merge(vec![rc, ra, rb, rcp])?;
        run_debug(&r)?;
        let inv_mapping = get_mapping::<f64>(&r)?;

        inv_mapping.into_iter().enumerate().for_each(|(indx, result)| {
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
            let c = 0 != indx & (1 << 2);
            let a = 0 != indx & (1 << 1);
            let b = 0 != indx & 1;

            let q_c = 0 != mapping & (1 << 2);
            let q_a = 0 != mapping & (1 << 1);
            let q_b = 0 != mapping & 1;

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

        let r = b.merge(vec![rc,ra,rb])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:04b}\t{:04b}", indx, mapping);
            let indx = indx as u64;
            let c = 0 != indx & (1 << 3);
            let a = 0 != indx & (1 << 2);
            let b = ((indx & (1 << 1)) >> 1) | ((indx & 1) << 1);

            let q_c = 0 != mapping & (1 << 3);
            let q_a = 0 != mapping & (1 << 2);
            let q_b = ((mapping & (1 << 1)) >> 1) | ((mapping & 1) << 1);

            let num = |x: bool| {
                if x { 1 } else { 0 }
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

        let r = b.merge(vec![rc,ra,rb])?;
        run_debug(&r)?;
        let mapping = get_mapping::<f64>(&r)?;

        mapping.into_iter().enumerate().for_each(|(indx, mapping)| {
            println!("{:07b}\t{:07b}", indx, mapping);
            let indx = indx as u64;
            let c = extract_bits(indx, &[5, 6]);
            let a = extract_bits(indx, &[3, 4]);
            let b = extract_bits(indx, &[0, 1, 2]);

            let q_c = extract_bits(mapping, &[5, 6]);
            let q_a = extract_bits(mapping, &[3, 4]);
            let q_b = extract_bits(mapping, &[0, 1, 2]);

            dbg!(c, a, b, q_c, q_a, q_b, (a + c + b) % (1 << 8));

            assert_eq!(q_c, c);
            assert_eq!(q_a, a);
            assert_eq!(q_b, (a + c + b) % (1 << 7));
        });
        Ok(())
    }
}