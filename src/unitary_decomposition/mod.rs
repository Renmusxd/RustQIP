/// Turn a unitary op into a series of gates.
pub mod circuit;

/// Decompose a unitary op into smaller controlled phases and rotations.
mod decomposition;
/// Utilities for unitary decomposition.
mod utils;
/// Find paths through series of bits via rotations.
mod bit_pathing;
