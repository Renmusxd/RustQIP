/// Turn a unitary op into a series of gates.
pub mod circuit;
/// Utilities for unitary decomposition.
pub mod utils;

/// Find paths through series of bits via rotations.
mod bit_pathing;
/// Decompose a unitary op into smaller controlled phases and rotations.
mod decomposition;

#[cfg(test)]
mod test_utils;
