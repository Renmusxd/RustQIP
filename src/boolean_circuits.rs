#[macro_use]
use crate::*;
use crate::{CircuitError, Register, UnitaryBuilder};

fn flip(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
    if ra.n() != rb.n() + 1 {
        CircuitError::make_err(format!("ra must have one more register than rb (current {} vs {})", ra.n(), rb.n()))
    } else {
        let m = ra.n();
        match (ra.n(), rb.n()) {
            (2, 1) => {
                // T|ra1>|ra0>|rb>
                Ok(b.cnot(ra, rb))
            },
            _ => {
                // T|am-1>|bm-3>|bm-2>
                let (ra, rb) = register_expr!(b, ra[m-1], rb[m-2, m-3]; |b, (ra, rb)| {
                    let (rb2, rb3) = b.split(rb, &[0])?;
                    let cr = b.merge(vec![ra, rb3]);

                    let (cr, rb2) = b.cnot(cr, rb2);

                    let (ra, rb3) = b.split(cr, &[0])?;
                    let rb = b.merge(vec![rb2, rb3]);
                    Ok((ra, rb))
                })?;

                // Flip |am-2..a0>|bm-3..b0>
                let ra_slice: Vec<_> = (0..m-2).collect();
                let rb_slice: Vec<_> = (0..m-3).collect();
                let (ra, rb) = register_expr!(b, ra ra_slice, rb rb_slice; |b, (ra, rb)| flip(b, ra, rb))?;

                // T|am-1>|bm-3>|bm-2>
                let (ra, rb) = register_expr!(b, ra[m-1], rb[m-2, m-3]; |b, (ra, rb)| {
                    let (rb2, rb3) = b.split(rb, &[0])?;
                    let cr = b.merge(vec![ra, rb3]);

                    let (cr, rb2) = b.cnot(cr, rb2);

                    let (ra, rb3) = b.split(cr, &[0])?;
                    let rb = b.merge(vec![rb2, rb3]);
                    Ok((ra, rb))
                })?;

                Ok((ra, rb))
            }
        }
    }
}
