/// Turn a unitary op into a series of gates.
pub mod circuit;

/// Find paths through series of bits via rotations.
mod bit_pathing;
/// Decompose a unitary op into smaller controlled phases and rotations.
mod decomposition;
/// Utilities for unitary decomposition.
mod utils;
