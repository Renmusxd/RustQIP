#[macro_use]
use crate::*;
use crate::{CircuitError, Register, UnitaryBuilder};

fn flip(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
    if ra.n() != rb.n() + 1 {
        CircuitError::make_err(format!("ra must have one more register than rb (current {} vs {})", ra.n(), rb.n()))
    } else {
        match (ra.n(), rb.n()) {
            (2, 1) => {
                Ok(b.cnot(ra, rb))
            },
            (na, nb) => {
                let (ra, rb) = register_expr!(b, ra[na-1], rb[nb-2, nb-3]; |b, (ra, rb)| {
                    let (rb2, rb3) = b.split(rb, &[0])?;
                    let cr = b.merge(vec![ra, rb3]);

                    let (cr, rb2) = b.cnot(cr, rb2);

                    let (ra, rb3) = b.split(cr, &[0])?;
                    let rb = b.merge(vec![rb2, rb3]);

                    Ok((ra, rb))
                })?;
                unimplemented!()
            }
        }
    }
}
