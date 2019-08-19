use crate::*;
use crate::{CircuitError, Register, UnitaryBuilder};

/// Perform an AND on all the qubits of ra, return ra alongside a single qubit qb with the result.
/// Uses a temporary qubit internally.
pub fn and(b: &mut OpBuilder, ra: Register) -> Result<(Register, Register), CircuitError> {
    let qb = b.qubit();
    match ra.n() {
        m if m <= 2 => {
            let (ra, qb) = b.cnot(ra, qb);
            Ok((ra, qb))
        },
        m => {
            let qt = b.get_temp_register(1, false);

            let m = m as usize;
            let k = m >> 1;
            let j = if m % 2 == 0 { k - 2 } else { k - 1 };
            dbg!(m, k, j);
            let (ra, qb, qt) = program!(b, ra, qb, qt;
                and_temp ra[k .. m-1], qt, ra[0 .. j];
                and_temp |qt, ra[0 .. j],| qb, ra[k .. k+j-2];
                and_temp ra[k .. m-1], qt, ra[0 .. j];
            );

            b.return_temp_register(qt, false);
            Ok((ra, qb))
        }
    }
}

fn cnot(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    let to_not = rs.pop().ok_or(CircuitError::new("Error unwrapping last qubit".to_string()))?;
    let cr = b.merge(rs)?;
    let (cr, to_not) = b.cnot(cr, to_not);
    Ok(vec![cr, to_not])
}

fn flip(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    let rb = rs.pop().ok_or(CircuitError::new("Error unwrapping rb".to_string()))?;
    let ra = rs.pop().ok_or(CircuitError::new("Error unwrapping ra".to_string()))?;

    let (ra, rb) = if ra.n() != rb.n() + 1 {
        CircuitError::make_err(format!("ra must have one more register than rb (current {} vs {})", ra.n(), rb.n()))
    } else {
        let m = ra.n() as usize;
        match (ra.n(), rb.n()) {
            (2, 1) => {
                // T|ra1>|ra0>|rb>
                Ok(b.cnot(ra, rb))
            },
            _ => {
                let (ra, rb) = program!(b, ra, rb;
                    cnot ra[m-1], rb[m-3], rb[m-2];
                    flip ra[0 .. m-2], rb[0 .. m-3];
                    cnot ra[m-1], rb[m-3], rb[m-2];
                );
                Ok((ra, rb))
            }
        }
    }?;
    Ok(vec![ra,rb])
}


fn and_temp(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    if rs.len() == 2 {
        let rb = rs.pop().ok_or(CircuitError::new("Error unwrapping rb".to_string()))?;
        let ra = rs.pop().ok_or(CircuitError::new("Error unwrapping ra".to_string()))?;
        let (ra, rb) = b.cnot(ra, rb);
        Ok(vec![ra, rb])
    } else {
        let rc = rs.pop().ok_or(CircuitError::new("Error unwrapping rc".to_string()))?;
        let rb = rs.pop().ok_or(CircuitError::new("Error unwrapping rb".to_string()))?;
        let ra = rs.pop().ok_or(CircuitError::new("Error unwrapping ra".to_string()))?;

        let m = ra.n();
        if rc.n() + 2 != m {
            CircuitError::make_err(format!("With m={} expected rc.n={} but found {}", m, m-2, rc.n()))
        } else if rb.n() != 1 {
            CircuitError::make_err(format!("Expected rb.n=1 found {}", rb.n()))
        } else {
            let (ra, rb, rc) = program!(b, ra, rb, rc;
                flip ra, |rb, rc,|;
                flip ra[0 .. ((ra.n() - 2) as usize)], rc;
            );
            Ok(vec![ra, rb, rc])
        }
    }
}

#[cfg(test)]
mod boolean_circuits_test {
    use super::*;

    #[test]
    fn test_and_basic() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let r = b.register(3)?;

        let (r,b) = and(&mut b, r)?;

        run_debug(&r)?;

        assert!(false);
        Ok(())
    }
}