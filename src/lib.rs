#![forbid(unsafe_code)]
#![deny(
    unreachable_pub,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

pub mod boolean_circuits;
pub mod builder;
pub mod builder_traits;
pub mod conditioning;
pub mod errors;
pub mod macros;
#[cfg(feature = "optimization")]
pub mod optimizer;
pub mod qfft;
pub mod rayon_helper;
pub mod state_ops;
pub mod types;
pub mod utils;

pub use num_complex::Complex;
pub use types::Precision;

pub mod prelude {
    pub use super::*;
    pub use crate::builder::LocalBuilder;
    pub use crate::builder_traits::*;
    pub use crate::conditioning::*;
    pub use crate::errors::*;
    pub use macros::RecursiveCircuitBuilder;
}
