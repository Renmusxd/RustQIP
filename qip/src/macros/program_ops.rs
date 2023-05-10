use crate::errors::CircuitResult;
use crate::prelude::{AdvancedCircuitBuilder, CliffordTBuilder};
use crate::types::Precision;

macro_rules! make_single_qubit_op {
    ($opname:ident, $funccall:ident) => {
        /// A $opname gate op.
        pub fn $opname<P: Precision, UB: CliffordTBuilder<P>>(
            b: &mut UB,
            r: UB::Register,
        ) -> CircuitResult<UB::Register> {
            Ok(b.$funccall(r))
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

/// A controlled NOT gate op.
pub fn cnot<P: Precision, UB: CliffordTBuilder<P>>(
    b: &mut UB,
    cr: UB::Register,
    r: UB::Register,
) -> CircuitResult<(UB::Register, UB::Register)> {
    b.cnot(cr, r)
}

/// A toffoli or generalized CNOT gate op.
pub fn toffoli<P: Precision, UB: AdvancedCircuitBuilder<P>>(
    b: &mut UB,
    cr: UB::Register,
    r: UB::Register,
) -> CircuitResult<(UB::Register, UB::Register)> {
    b.toffoli(cr, r)
}
