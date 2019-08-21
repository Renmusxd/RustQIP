use crate::*;

fn carry(
    b: &mut dyn UnitaryBuilder,
    ra: Register,
    rb: Register,
    rc: Register,
    rcp: Register,
) -> Result<(Register, Register, Register, Register), CircuitError> {
    let x_op = |b: &mut dyn UnitaryBuilder,
                mut rs: Vec<Register>|
     -> Result<Vec<Register>, CircuitError> {
        let ra = rs
            .pop()
            .ok_or_else(|| CircuitError::new("Error unwrapping rc for carry".to_string()))?;
        let ra = b.x(ra);
        Ok(vec![ra])
    };

    let (ra, rb, rc, rcp) = program!(b, ra, rb, rc, rcp;
        control x_op |ra, rb,| rcp;
        control x_op ra, rb;
        control x_op |rc, rb,| rcp;
        control x_op ra, rb;
    )?;

    Ok((ra, rb, rc, rcp))
}
