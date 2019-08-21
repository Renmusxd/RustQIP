use crate::*;

/// Add together ra and rb using rc as carry, result is in rb.
pub fn add(b: &mut dyn UnitaryBuilder, rc: Register, ra: Register, rb: Register) -> Result<(Register, Register, Register), CircuitError> {
    wrap_fn!(carry_op, carry?, ra, rb, rc, rcp);
    wrap_fn!(sum_op, sum, ra, rb, rc);
    wrap_fn!(add_op, add?, ra, rb, rc);

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
                // TODO inverse carry
                sum_op rc[0], ra[0], rb[0];
            )?;
            Ok((rc, ra, rb))
        },
        (nc, na, nb) => CircuitError::make_err(format!("Expected rc[n] ra[n] and rb[n+1], but got ({},{},{})", nc, na, nb))
    }
}

fn sum(b: &mut dyn UnitaryBuilder,
       ra: Register,
       rb: Register,
       rc: Register) -> (Register, Register, Register) {
    let (ra, rb) = b.cx(ra, rb);
    let (rc, rb) = b.cx(rc, rb);
    (ra, rb, rc)
}

fn carry(
    b: &mut dyn UnitaryBuilder,
    ra: Register,
    rb: Register,
    rc: Register,
    rcp: Register,
) -> Result<(Register, Register, Register, Register), CircuitError> {
    fn x(b: &mut dyn UnitaryBuilder, r: Register) -> Register {
        b.x(r)
    }
    wrap_fn!(x_op, x, r);

    let (ra, rb, rc, rcp) = program!(b, ra, rb, rc, rcp;
        control x_op |ra, rb,| rcp;
        control x_op ra, rb;
        control x_op |rc, rb,| rcp;
        control x_op ra, rb;
    )?;

    Ok((ra, rb, rc, rcp))
}