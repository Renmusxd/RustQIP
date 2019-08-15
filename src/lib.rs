#![deny(
    missing_docs,
    unreachable_pub,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

//! Quantum Computing library leveraging graph building to build efficient quantum circuit
//! simulations.
//!
//! See all the examples in the [examples directory](https://github.com/Renmusxd/RustQIP/tree/master/examples) of the Github repository.
//!
//! # Example (CSWAP)
//! Here's an example of a small circuit where two groups of Registers are swapped conditioned on a
//! third. This circuit is very small, only three operations plus a measurement, so the boilerplate
//! can look quite large in compairison, but that setup provides the ability to construct circuits
//! easily and safely when they do get larger.
//! ```
//! use qip::*;
//!
//! # fn main() -> Result<(), CircuitError> {
//! // Make a new circuit builder.
//! let mut b = OpBuilder::new();
//!
//! // Make three registers of sizes 1, 3, 3 (7 qubits total).
//! let q = b.qubit();
//! let ra = b.register(3)?;
//! let rb = b.register(3)?;
//!
//! // We will want to feed in some inputs later, hang on to the handles
//! // so we don't need to actually remember any indices.
//! let a_handle = ra.handle();
//! let b_handle = rb.handle();
//!
//! // Define circuit
//! // First apply an H to r
//! let q = b.hadamard(q);
//! // Then run this subcircuit conditioned on r, applied to ra and rb
//! let (q, _, _) = b.cswap(q, ra, rb)?;
//! // Finally apply H to q again.
//! let q = b.hadamard(q);
//!
//! // Add a measurement to the first qubit, save a reference so we can get the result later.
//! let (q, m_handle) = b.measure(q);
//!
//! // Now q is the end result of the above circuit, and we can run the circuit by referencing it.
//!
//! // Make an initial state: |0,000,001>
//! let initial_state = [a_handle.make_init_from_index(0)?,
//!                      b_handle.make_init_from_index(1)?];
//! // Run circuit with a given precision.
//! let (_, measured) = run_local_with_init::<f64>(&q, &initial_state)?;
//!
//! // Lookup the result of the measurement we performed using the handle.
//! let (result, p) = measured.get_measurement(&m_handle).unwrap();
//!
//! // Print the measured result
//! println!("Measured: {:?} (with chance {:?})", result, p);
//! # Ok(())
//! # }
//! ```
//!

pub use self::builders::{OpBuilder, UnitaryBuilder};
pub use self::common_circuits::*;
pub use self::errors::*;
pub use self::pipeline::{run_local, run_local_with_init, run_with_state, QuantumState};
pub use self::pipeline_debug::run_debug;
pub use self::qubit_chainer::{chain, chain_tuple, chain_vec};
pub use self::qubits::Register;
pub use self::types::Precision;
pub use num::Complex;

/// Opbuilder and such
pub mod builders;
/// Common circuits for general usage.
pub mod common_circuits;
/// Error values for the library.
pub mod errors;
/// Code for building pipelines.
pub mod pipeline;
/// Tools for displaying pipelines.
pub mod pipeline_debug;
/// Quantum fourier transform support.
pub mod qfft;
/// Ease of use for chains of single register ops.
pub mod qubit_chainer;
/// Basic classes for defining circuits/pipelines.
pub mod qubits;
/// Commonly used types.
pub mod types;
/// Break unitary matrices into circuits.
pub mod unitary_decomposition;
/// Commonly used short functions.
pub mod utils;

/// Efficient iterators for sparse kronprod matrices.
pub mod iterators;
/// Functions for measuring states.
pub mod measurement_ops;
/// Functions for running ops on states.
pub mod state_ops;
