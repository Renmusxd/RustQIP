/// Main mod with program and macros, all reexported.
#[macro_use]
pub mod program;

/// Common ops for programs, not automatically exported because their names easily conflict with
/// existing variables.
pub mod common_ops;
/// Tools for inverting functions on qubits.
#[macro_use]
pub mod inverter;

pub use inverter::*;
pub use program::*;
