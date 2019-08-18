#[macro_use]
use crate::*;
use crate::{CircuitError, Register, UnitaryBuilder};

fn cnot(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    let to_not = rs.pop().ok_or(CircuitError::new("Error unwrapping last qubit".to_string()))?;
    let cr = b.merge(rs);
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
                let ra_range: Vec<_> = (0 .. m-2).collect();
                let rb_range: Vec<_> = (0 .. m-3).collect();
                let (ra, rb) = program!(b, ra, rb;
                    cnot ra[m-1], rb[m-3], rb[m-2];
                    flip ra (ra_range), rb (rb_range);
                    cnot ra[m-1], rb[m-3], rb[m-2];
                );
                Ok((ra, rb))
            }
        }
    }?;
    Ok(vec![ra,rb])
}

//fn and_temp(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
//    let rc = rs.pop().ok_or(CircuitError::new("Error unwrapping rc".to_string()))?;
//    let rb = rs.pop().ok_or(CircuitError::new("Error unwrapping rb".to_string()))?;
//    let ra = rs.pop().ok_or(CircuitError::new("Error unwrapping ra".to_string()))?;
//    let m = ra.n();
//    let rb_indices = rb.indices.clone();
//    let r = b.merge(vec![rb, rc]);
//    let mut rs = flip(b, vec![ra, r])?;
//    let r = rs.pop().unwrap();
//    let ra = rs.pop().unwrap();
//    let (rb, rc) = b.split(r, &rb_indices)?;
//
//    let ra_slice: Vec<_> = (0..m-2).collect();
//    let (ra, rc) = register_expr!(b, ra ra_slice, rc; |b, (ra, rc)| flip(b, ra, rc))?;
//    Ok((ra, rb, rc))
//}
//
//fn and(b: &mut OpBuilder, ra: Register) -> Result<(Register, Register), CircuitError> {
//    let m = ra.n();
//    let k = m >> 1;
//    let j = k - 2;
//
//    if m <= 2 {
//        let rb = b.qubit();
//        Ok(b.cnot(ra, rb))
//    } else {
//        let temp_q = b.get_temp_register(1, false);
//
//        let front_slice: Vec<_> = (k .. m-1).collect();
//        let (ra_front, ra) = b.split(ra, &front_slice)?;
//        let back_slice: Vec<_> = (0 .. j).collect();
//        let (ra_back, ra) = b.split(ra, &back_slice)?;
//        let (ra_front, temp_q, ra_back) = and_temp(b, ra_front, temp_q, ra_back)?;
//        let ra_backs = b.split_all(ra_back);
//        let ra = b.merge_with_indices(ra, ra_backs, &back_slice)?;
//        let ra_fronts = b.split_all(ra_front);
//        let ra = b.merge_with_indices(ra, ra_fronts, &front_slice)?;
//
//        let front_register = b.merge(vec![ra_back, temp_q]);
//        let back_slice: Vec<_> = (k .. k+j-2).collect();
//        let (back_register, remaining) =
//
//        b.return_temp_register(temp_q, false);
//    }
//}