use crate::errors::{CircuitError, CircuitResult};
use crate::prelude::{AdvancedCircuitBuilder, CliffordTBuilder};
use crate::Precision;

macro_rules! make_single_qubit_op {
    ($opname:ident, $funccall:ident) => {
        pub fn $opname<P: Precision, UB: CliffordTBuilder<P>>(
            b: &mut UB,
            r: Vec<UB::Register>,
        ) -> CircuitResult<Vec<UB::Register>> {
            Ok(b.merge_registers(r.into_iter())
                .map(|r| b.$funccall(r))
                .map(|r| vec![r])
                .unwrap_or(vec![]))
        }
    };
}

make_single_qubit_op!(not, not);
make_single_qubit_op!(h, h);
make_single_qubit_op!(x, x);
make_single_qubit_op!(y, y);
make_single_qubit_op!(z, z);
make_single_qubit_op!(s, s);
make_single_qubit_op!(t, t);

pub fn cnot<P: Precision, UB: CliffordTBuilder<P>>(
    b: &mut UB,
    mut rs: Vec<UB::Register>,
) -> CircuitResult<Vec<UB::Register>> {
    let r = rs
        .pop()
        .ok_or_else(|| CircuitError::new("CNOT requires at least 2 qubits."))?;
    let cr = b
        .merge_registers(rs.into_iter())
        .ok_or_else(|| CircuitError::new("CNOT requires at least 2 qubits."))?;
    let (cr, r) = b.cnot(cr, r)?;
    let r = b.merge_two_registers(cr, r);
    Ok(vec![r])
}

pub fn toffoli<P: Precision, UB: AdvancedCircuitBuilder<P>>(
    b: &mut UB,
    mut rs: Vec<UB::Register>,
) -> CircuitResult<Vec<UB::Register>> {
    let r = rs
        .pop()
        .ok_or_else(|| CircuitError::new("Toffoli requires at least 2 qubits."))?;
    let cr = b
        .merge_registers(rs.into_iter())
        .ok_or_else(|| CircuitError::new("Toffoli requires at least 2 qubits."))?;
    let (cr, r) = b.toffoli(cr, r)?;
    let r = b.merge_two_registers(cr, r);
    Ok(vec![r])
}
