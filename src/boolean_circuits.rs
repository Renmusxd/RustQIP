#[macro_use]
use crate::*;
use crate::{CircuitError, Register, UnitaryBuilder};

//fn flip(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
//    if ra.n() != rb.n() + 1 {
//        CircuitError::make_err(format!("ra must have one more register than rb (current {} vs {})", ra.n(), rb.n()))
//    } else {
//        let m = ra.n();
//        match (ra.n(), rb.n()) {
//            (2, 1) => {
//                // T|ra1>|ra0>|rb>
//                Ok(b.cnot(ra, rb))
//            },
//            _ => {
//                // T|am-1>|bm-3>|bm-2>
//                let (ra, rb) = register_expr!(b, ra[m-1], rb[m-2, m-3]; |b, (ra, rb)| {
//                    let (rb2, rb3) = b.split(rb, &[0])?;
//                    let cr = b.merge(vec![ra, rb3]);
//
//                    let (cr, rb2) = b.cnot(cr, rb2);
//
//                    let (ra, rb3) = b.split(cr, &[0])?;
//                    let rb = b.merge(vec![rb2, rb3]);
//                    Ok((ra, rb))
//                })?;
//
//                // Flip |am-2..a0>|bm-3..b0>
//                let ra_slice: Vec<_> = (0..m-2).collect();
//                let rb_slice: Vec<_> = (0..m-3).collect();
//                let (ra, rb) = register_expr!(b, ra ra_slice, rb rb_slice; |b, (ra, rb)| flip(b, ra, rb))?;
//
//                // T|am-1>|bm-3>|bm-2>
//                let (ra, rb) = register_expr!(b, ra[m-1], rb[m-2, m-3]; |b, (ra, rb)| {
//                    let (rb2, rb3) = b.split(rb, &[0])?;
//                    let cr = b.merge(vec![ra, rb3]);
//
//                    let (cr, rb2) = b.cnot(cr, rb2);
//
//                    let (ra, rb3) = b.split(cr, &[0])?;
//                    let rb = b.merge(vec![rb2, rb3]);
//                    Ok((ra, rb))
//                })?;
//
//                Ok((ra, rb))
//            }
//        }
//    }
//}
//
//fn and_temp(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register, rc: Register) -> Result<(Register, Register, Register), CircuitError> {
//    let m = ra.n();
//    let rb_indices = rb.indices.clone();
//    let r = b.merge(vec![rb, rc]);
//    let (ra, r) = flip(b, ra, r)?;
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