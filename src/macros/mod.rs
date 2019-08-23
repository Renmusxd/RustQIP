/// Main mod with program and macros, all reexported.
#[macro_use]
pub mod program;

/// Common ops for programs, not automatically exported because their names easily conflict with
/// existing variables.
pub mod common_ops;
mod register_expression;

pub use program::*;
