/// Main mod with program and macros, all reexported.
#[macro_use]
pub mod program;

mod register_expression;
/// Common ops for programs, not automatically exported because their names easily conflict with
/// existing variables.
pub mod common_ops;

pub use program::*;
