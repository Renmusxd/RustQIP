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
//! can look quite large in comparison, but that setup provides the ability to construct circuits
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
//! # The Program Macro
//! While the borrow checker included in rust is a wonderful tool for checking that our registers
//! are behaving, it can be cumbersome. For that reason qip also includes a macro which provides an
//! API similar to that which you would see in quantum computing textbooks
//! ```
//! use qip::*;
//! # fn main() -> Result<(), CircuitError> {
//!
//! let n = 3;
//! let mut b = OpBuilder::new();
//! let ra = b.register(n)?;
//! let rb = b.register(n)?;
//!
//! let gamma = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
//!     let rb = rs.pop().unwrap();
//!     let ra = rs.pop().unwrap();
//!     let (ra, rb) = b.cnot(ra, rb);
//!     let (rb, ra) = b.cnot(rb, ra);
//!     Ok(vec![ra, rb])
//! };
//!
//! let (ra, rb) = program!(&mut b, ra, rb;
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]>
//!     gamma ra[0..2], ra[2];
//!     // Applies gamma to |ra[0] rb[0]>|ra[2]>
//!     gamma |ra[0], rb[0],| ra[2];
//!     // Applies gamma to |ra[0]>|rb[0] ra[2]>
//!     gamma ra[0], |rb[0], ra[2],|;
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |111>
//!     control gamma rb, ra[0..2], ra[2];
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |110> (rb[0] == |0>, rb[1] == 1, ...)
//!     control(0b110) gamma rb, ra[0..2], ra[2];
//! )?;
//! let r = b.merge(vec![ra, rb])?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! To clean up gamma we can use the `wrap_fn` macro:
//!
//! ```
//! use qip::*;
//! # fn main() -> Result<(), CircuitError> {
//!
//! let n = 3;
//! let mut b = OpBuilder::new();
//! let ra = b.register(n)?;
//! let rb = b.register(n)?;
//!
//! fn gamma(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> (Register, Register) {
//!     let (ra, rb) = b.cnot(ra, rb);
//!     let (rb, ra) = b.cnot(rb, ra);
//!     (ra, rb)
//! }
//! wrap_fn!(gamma_op, gamma, ra, rb); // if gamma returns a Result, write (gamma) instead.
//!
//! let (ra, rb) = program!(&mut b, ra, rb;
//!     gamma_op ra[0..2], ra[2];
//! )?;
//! let r = b.merge(vec![ra, rb])?;
//!
//! # Ok(())
//! # }
//! ```

pub use self::builders::*;
pub use self::common_circuits::*;
pub use self::errors::*;
pub use self::macros::*;
pub use self::pipeline::{run_local, run_local_with_init, run_with_state, QuantumState};
pub use self::pipeline_debug::run_debug;
pub use self::qubits::Register;
pub use self::types::Precision;
pub use num::Complex;

/// Quantum analogues of boolean circuits
pub mod boolean_circuits;
/// Opbuilder and such
pub mod builders;
/// Common circuits for general usage.
pub mod common_circuits;
/// Error values for the library.
pub mod errors;
/// Macros for general ease of use.
#[macro_use]
pub mod macros;
/// Efficient iterators for sparse kronprod matrices.
pub mod iterators;
/// Functions for measuring states.
pub mod measurement_ops;
/// Code for building pipelines.
pub mod pipeline;
/// Tools for displaying pipelines.
pub mod pipeline_debug;
/// Quantum fourier transform support.
pub mod qfft;
/// Basic classes for defining circuits/pipelines.
pub mod qubits;
/// Sparse quantum states
pub mod sparse_state;
/// Functions for running ops on states.
pub mod state_ops;
/// Tracing state
pub mod trace_state;
/// Commonly used types.
pub mod types;
/// Break unitary matrices into circuits.
pub mod unitary_decomposition;
/// Commonly used short functions.
pub mod utils;
