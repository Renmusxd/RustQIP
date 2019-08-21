use crate::*;

fn carry(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    let rcp = rs.pop().ok_or_else(|| CircuitError::new("Error unwrapping rcp for carry".to_string()))?;
    let rc = rs.pop().ok_or_else(|| CircuitError::new("Error unwrapping rc for carry".to_string()))?;
    let rb = rs.pop().ok_or_else(|| CircuitError::new("Error unwrapping rb for carry".to_string()))?;
    let ra = rs.pop().ok_or_else(|| CircuitError::new("Error unwrapping ra for carry".to_string()))?;

    let x_op = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
        let ra = rs.pop().ok_or_else(|| CircuitError::new("Error unwrapping rc for carry".to_string()))?;
        let ra = b.x(ra);
        Ok(vec![ra])
    };

    let (ra, rb, rc, rcp) = program!(b, ra, rb, rc, rcp;
        control x_op |ra, rb,| rcp;
        control x_op ra, rb;
        control x_op |rc, rb,| rcp;
        control x_op ra, rb;
    );

    Ok(vec![ra,rb,rc,rcp])
}