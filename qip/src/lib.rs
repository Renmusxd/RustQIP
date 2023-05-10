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
//! ```
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//! # #[cfg(feature = "macros")]
//! use qip_macros::program;
//! # #[cfg(feature = "macros")]
//! # fn main() -> CircuitResult<()> {
//!
//! fn gamma<B>(b: &mut B, ra: B::Register, rb: B::Register) -> CircuitResult<(B::Register, B::Register)>
//!    where B: AdvancedCircuitBuilder<f64>
//! {
//!     let (ra, rb) = b.toffoli(ra, rb)?;
//!     let (rb, ra) = b.toffoli(rb, ra)?;
//!     Ok((ra, rb))
//! }
//!
//! let n = NonZeroUsize::new(3).unwrap();
//! let mut b = LocalBuilder::default();
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//!
//! let (ra, rb) = program!(&mut b; ra, rb;
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]>
//!     gamma ra[0..2], ra[2];
//!     // Applies gamma to |ra[0] rb[0]>|ra[2]>
//!     // Notice ra[0] and rb[0] are grouped by brackets.
//!     gamma [ra[0], rb[0]], ra[2];
//!     // Applies gamma to |ra[0]>|rb[0] ra[2]>
//!     gamma ra[0], [rb[0], ra[2]];
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |111>
//!     control gamma rb, ra[0..2], ra[2];
//!     // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |110> (rb[0] == |0>, rb[1] == 1, ...)
//!     control(0b110) gamma rb, ra[0..2], ra[2];
//! )?;
//! let r = b.merge_two_registers(ra, rb);
//!
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "macros"))]
//! # fn main() {}
//! ```
//!
//! We can also apply this to function which take other argument. Here `gamma` takes a boolean
//! argument `skip` which is passed in before the registers.
//! *The arguments to functions in the program macro may not reference the input registers*
//! ```rust
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//! # #[cfg(feature = "macros")]
//! use qip_macros::program;
//! # #[cfg(feature = "macros")]
//! # fn main() -> CircuitResult<()> {
//!
//! fn gamma<B>(b: &mut B, skip: bool, ra: B::Register, rb: B::Register) -> CircuitResult<(B::Register, B::Register)>
//!    where B: AdvancedCircuitBuilder<f64>
//! {
//!     let (ra, rb) = b.toffoli(ra, rb)?;
//!     let (rb, ra) = if skip {
//!         b.toffoli(rb, ra)?
//!     } else {
//!         (rb, ra)
//!     };
//!     Ok((ra, rb))
//! }
//!
//! let n = NonZeroUsize::new(3).unwrap();
//! let mut b = LocalBuilder::default();
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//! let (ra, rb) = program!(&mut b; ra, rb;
//!     gamma(true) ra[0..2], ra[2];
//!     gamma(0 == 1) ra[0..2], ra[2];
//! )?;
//! let r = b.merge_two_registers(ra, rb);
//!
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "macros"))]
//! # fn main() {}
//! ```
//!
//! # The Invert Macro
//! It's often useful to define functions of registers as well as their inverses, the `#[invert]`
//! macro automates much of this process.
//! ```
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//! # #[cfg(feature = "macros")]
//! use qip_macros::*;
//! # #[cfg(feature = "macros")]
//! # fn main() -> CircuitResult<()> {
//!
//! use qip::inverter::Invertable;
//! // Make gamma and its inverse: gamma_inv
//! #[invert(gamma_inv)]
//! fn gamma<B>(b: &mut B, ra: B::Register, rb: B::Register) -> CircuitResult<(B::Register, B::Register)>
//!    where B: AdvancedCircuitBuilder<f64> + Invertable<SimilarBuilder=B>
//! {
//!     let (ra, rb) = b.toffoli(ra, rb)?;
//!     let (rb, ra) = b.toffoli(rb, ra)?;
//!     Ok((ra, rb))
//! }
//!
//! let n = NonZeroUsize::new(3).unwrap();
//! let mut b = LocalBuilder::default();
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//!
//! let (ra, rb) = program!(&mut b; ra, rb;
//!     gamma ra[0..2], ra[2];
//!     gamma_inv ra[0..2], ra[2];
//! )?;
//! let r = b.merge_two_registers(ra, rb);
//!
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "macros"))]
//! # fn main() {}
//! ```
//!
//! To invert functions with additional arguments, we must list the non-register arguments.
//! ```
//! use qip::prelude::*;
//! use std::num::NonZeroUsize;
//! # #[cfg(feature = "macros")]
//! use qip_macros::*;
//! # #[cfg(feature = "macros")]
//! # fn main() -> CircuitResult<()> {
//!
//! use qip::inverter::Invertable;
//! // Make gamma and its inverse: gamma_inv
//! #[invert(gamma_inv, skip)]
//! fn gamma<B>(b: &mut B, skip: bool, ra: B::Register, rb: B::Register) -> CircuitResult<(B::Register, B::Register)>
//!    where B: AdvancedCircuitBuilder<f64> + Invertable<SimilarBuilder=B>
//! {
//!     let (ra, rb) = b.toffoli(ra, rb)?;
//!     let (rb, ra) = if skip {
//!         b.toffoli(rb, ra)?
//!     } else {
//!         (rb, ra)
//!     };
//!     Ok((ra, rb))
//! }
//!
//! let n = NonZeroUsize::new(3).unwrap();
//! let mut b = LocalBuilder::default();
//! let ra = b.register(n);
//! let rb = b.register(n);
//!
//! let (ra, rb) = program!(&mut b; ra, rb;
//!     gamma(true) ra[0..2], ra[2];
//!     gamma_inv(true) ra[0..2], ra[2];
//! )?;
//! let r = b.merge_two_registers(ra, rb);
//!
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "macros"))]
//! # fn main() {}
//! ```

/// Quantum analog of reversible classical circuits, such as `and`, `or`, `add`, and `multiply`
#[cfg(feature = "macros")]
pub mod boolean_circuits;
/// A circuit builder implementation which builds circuits out of simple elements.
pub mod builder;
/// Standard traits for circuit builders.
pub mod builder_traits;
/// Traits for constructing conditioned circuit builders.
pub mod conditioning;
/// Circuit builder error types.
pub mod errors;
/// Functions and traits for inverting circuits.
pub mod inverter;
/// Types for helping procedural macros.
#[cfg(feature = "macros")]
pub mod macros;
/// Standard quantum fourier transform implementation.
pub mod qfft;
/// Lower-level circuit operations.
pub mod state_ops;
/// Reusable types.
pub mod types;
/// Utility functions for bit and index manipulation
pub mod utils;

pub use num_complex::Complex;
pub use rand;
pub use types::*;

/// Commonly used types and traits.
/// ```
/// use qip::prelude::*;
/// ```
pub mod prelude {
    pub use super::*;
    pub use crate::builder::LocalBuilder;
    pub use crate::builder_traits::*;
    pub use crate::conditioning::*;
    pub use crate::errors::*;
    pub use crate::inverter::RecursiveCircuitBuilder;
}
