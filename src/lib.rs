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

//! Quantum Computing library leveraging graph building to build efficient quantum circuit
//! simulations.
//! Rust is a great language for quantum computing with gate models because the borrow checker
//! is very similar to the [No-cloning theorem](https://wikipedia.org/wiki/No-cloning_theorem).
//!
//! See all the examples in the [examples directory](https://github.com/Renmusxd/RustQIP/tree/master/examples) of the Github repository.
//!
//! # Example (CSWAP)
//! Here's an example of a small circuit where two groups of Registers are swapped conditioned on a
//! third. This circuit is very small, only three operations plus a measurement, so the boilerplate
//! can look quite large in comparison, but that setup provides the ability to construct circuits
//! easily and safely when they do get larger.
//! ```
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//!
//! # fn main() -> CircuitResult<()> {
//! // Make a new circuit builder.
//! let mut b = LocalBuilder::<f64>::default();
//! let n = NonZeroUsize::new(3).unwrap();
//!
//! // Make three registers of sizes 1, 3, 3 (7 qubits total).
//! let q = b.qubit();  // Same as b.register(1)?;
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//! // Define circuit
//! // First apply an H to q
//! let q = b.h(q);
//! // Then swap ra and rb, conditioned on q.
//! let mut cb = b.condition_with(q);
//! let (ra, rb) = cb.swap(ra, rb)?;
//! let q = cb.dissolve();
//! // Finally apply H to q again.
//! let q = b.h(q);
//!
//! // Add a measurement to the first qubit, save a reference so we can get the result later.
//! let (q, m_handle) = b.measure(q);
//!
//! // Now q is the end result of the above circuit, and we can run the circuit by referencing it.
//!
//! // Run circuit with a given precision.
//! let (_, measured) = b.calculate_state_with_init([(&ra, 0b000), (&rb, 0b001)]);
//!
//! // Lookup the result of the measurement we performed using the handle, and the probability
//! // of getting that measurement.
//! let (result, p) = measured.get_measurement(m_handle);
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
//! API similar to that which you would see in quantum computing textbooks.
//! *Notice that due to a design choice in rust's `macro_rules!` we use vertical bars to group qubits
//! and a comma must appear before the closing bar. This may be fixed in the future using procedural
//! macros.*
//! ```
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//! # fn main() -> CircuitResult<()> {
//!
//! let n = NonZeroUsize::new(3).unwrap();
//! let mut b = LocalBuilder::default();
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//! fn gamma<B>(b: &mut B, mut rs: Vec<B::Register>) -> CircuitResult<Vec<B::Register>>
//!    where B: AdvancedCircuitBuilder<f64>
//! {
//!     let rb = rs.pop().ok_or(CircuitError::new("No rb provided"))?;
//!     let ra = rs.pop().ok_or(CircuitError::new("No ra provided"))?;
//!     let (ra, rb) = b.toffoli(ra, rb)?;
//!     let (rb, ra) = b.toffoli(rb, ra)?;
//!     Ok(vec![ra, rb])
//! }
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
//! let r = b.merge_two_registers(ra, rb);
//!
//! # Ok(())
//! # }
//! ```

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
pub use rand;
pub use types::Precision;

pub mod prelude {
    pub use super::*;
    pub use crate::builder::LocalBuilder;
    pub use crate::builder_traits::*;
    pub use crate::conditioning::*;
    pub use crate::errors::*;
    pub use macros::RecursiveCircuitBuilder;
}
