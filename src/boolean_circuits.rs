use crate::common_circuits::*;
use crate::{CircuitError, Register, UnitaryBuilder};

//fn flip(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
//    if ra.n() != rb.n() + 1 {
//        CircuitError::make_err(format!("ra must have one more register than rb (current {} vs {})", ra.n(), rb.n()))
//    } else {
//        match (ra.n(), rb.n()) {
//            (2, 1) => {
//                Ok(b.cnot(ra, rb))
//            },
//            (na, nb) => {
//                let (a_m1, ra) = b.split(ra, &[na-1])?;
//                let (b_m2, rb) = b.split(rb, &[nb-2])?;
//                let (b_m3, rb) = b.split(rb, &[nb-3])?;
//                let cr = b.merge(vec![a_m1, b_m3]);
//                let (cr, b_m2) = b.cnot(cr, b_m2);
//                let (a_m1, b_m3) = b.split(cr, &[0])?;
//
//
//            }
//        }
//    }
//}
