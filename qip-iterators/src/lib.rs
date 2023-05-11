#![forbid(unsafe_code)]
#![deny(
    unreachable_pub,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unstable_features,
    unused_import_braces,
    unused_qualifications,
    missing_docs
)]

//! Matrix multiplication library for qubit-structure matrices.

/// Iterators over nonzero elements of tensor-product matrices.
pub mod iterators;
/// Helper functions for using iterators for matrix multiplication
pub mod matrix_ops;
/// Helpers for converting from synchronous to parallel iterators.
pub mod rayon_helper;
/// Utilities related to bit manipulation of tensor elements.
pub mod utils;
