use crate::pipeline::StateModifier;
use crate::{Register, UnitaryBuilder};

/// A gateset converter for circuits.
pub trait GateSet<U: UnitaryBuilder> {
    /// Make a new gateset which builds using a unitary builder
    fn new(b: U) -> Self;
    /// Feed statemodifiers to the gateset converter
    fn feed(&mut self, u: &StateModifier);
    /// Dissolve the gateset and retrieve the builder and registers.
    fn dissolve(self) -> (U, Register);
}
