use num_traits::{Float, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};

/// The float precision of the circuit.
pub trait Precision:
    Default + NumAssign + Float + Sum + Send + Sync + Display + Product + Debug
{
}

impl Precision for f64 {}

impl Precision for f32 {}

/// Order of qubits returned by `QuantumState::into_state` and other similar methods.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Representation {
    /// Qubit with index 0 is the least significant index bit.
    LittleEndian,
    /// Qubit with index 0 is the most significant index bit.
    BigEndian,
}
