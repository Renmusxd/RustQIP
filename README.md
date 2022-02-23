# RustQIP

Quantum Computing library leveraging graph building to build efficient quantum circuit simulations.

[![qip on crates.io](https://img.shields.io/crates/v/qip.svg)](https://crates.io/crates/qip)
[![qip docs](https://img.shields.io/badge/docs-docs.rs-orange.svg)](https://docs.rs/qip)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

See all the examples in the [examples directory](https://github.com/Renmusxd/RustQIP/tree/master/examples).

*PRs welcome*

_Note: Currently undergoing a large rewrite, the examples below are valid for the crates.io versions of the library but
are out of date for the bleeding edge_

# Example (CSWAP)

Here's an example of a small circuit where two groups of Registers are swapped conditioned on a third. This circuit is
very small, only three operations plus a measurement, so the boilerplate can look quite large in comparison, but that
setup provides the ability to construct circuits easily and safely when they do get larger.

```rust
use qip::*;

// Make a new circuit builder.
let mut b = OpBuilder::new();

// Make three registers of sizes 1, 3, 3 (7 qubits total).
let q = b.qubit();  // Same as b.register(1)?;
let ra = b.register(3) ?;
let rb = b.register(3) ?;

// We will want to feed in some inputs later, hang on to the handles
// so we don't need to actually remember any indices.
let a_handle = ra.handle();
let b_handle = rb.handle();

// Define circuit
// First apply an H to q
let q = b.hadamard(q);
// Then swap ra and rb, conditioned on q.
let (q, _, _) = b.cswap(q, ra, rb) ?;
// Finally apply H to q again.
let q = b.hadamard(q);
// Add a measurement to the first qubit, save a reference so we can get the result later.
let (q, m_handle) = b.measure(q);

// Now q is the end result of the above circuit, and we can run the circuit by referencing it.
// Make an initial state: |0,000,001> (default value for registers not mentioned is 0).
let initial_state = [a_handle.make_init_from_index(0b000) ?,
b_handle.make_init_from_index(0b001) ? ];

// Run circuit with a given precision.
let (_, measured) = run_local_with_init::<f64>( & q, & initial_state) ?;

// Lookup the result of the measurement we performed using the handle, and the probability
// of getting that measurement.
let (result, p) = measured.get_measurement( & m_handle).unwrap();
println!("Measured: {:?} (with chance {:?})", result, p);
```

# The Program Macro

While the borrow checker included in rust is a wonderful tool for checking that our registers are behaving, it can be
cumbersome. For that reason qip also includes a macro which provides an API similar to that which you would see in
quantum computing textbooks.
*Notice that due to a design choice in rust's `macro_rules!` we use vertical bars to group qubits and a comma must
appear before the closing bar. This may be fixed in the future using procedural macros.*

```rust
use qip::*;

let n = 3;
let mut b = OpBuilder::new();
let ra = b.register(n) ?;
let rb = b.register(n) ?;

fn gamma(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
    let rb = rs.pop().unwrap();
    let ra = rs.pop().unwrap();
    let (ra, rb) = b.cnot(ra, rb);
    let (rb, ra) = b.cnot(rb, ra);
    Ok(vec![ra, rb])
}

let (ra, rb) = program!(&mut b, ra, rb;
    // Applies gamma to |ra[0] ra[1]>|ra[2]>
    gamma ra[0..2], ra[2];
    // Applies gamma to |ra[0] rb[0]>|ra[2]>
    gamma |ra[0], rb[0],| ra[2];
    // Applies gamma to |ra[0]>|rb[0] ra[2]>
    gamma ra[0], |rb[0], ra[2],|;
    // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |111>
    control gamma rb, ra[0..2], ra[2];
    // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |110> (rb[0] == |0>, rb[1] == 1, ...)
    control(0b110) gamma rb, ra[0..2], ra[2];
)?;
let r = b.merge(vec![ra, rb]) ?;
```

To clean up gamma we can use the `wrap_fn` macro:

```rust
use qip::*;

let n = 3;
let mut b = OpBuilder::new();
let ra = b.register(n) ?;
let rb = b.register(n) ?;

fn gamma(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> (Register, Register) {
    let (ra, rb) = b.cnot(ra, rb);
    let (rb, ra) = b.cnot(rb, ra);
    (ra, rb)
}

// Make a function gamma_op from gamma which matches the spec required by program!(...).
// Here we tell wrap_fn! that gamma takes two registers, which we will internally call ra, rb.
// if gamma returns a Result<(Register, Register), CircuitError>, write (gamma) instead.
// wrap_fn!(gamma_op, (gamma), ra, rb)
wrap_fn!(gamma_op, gamma, ra, rb);

let (ra, rb) = program!(&mut b, ra, rb;
    gamma_op ra[0..2], ra[2];
)?;
let r = b.merge(vec![ra, rb]) ?;
```

And with these wrapped functions, automatically produce their conjugates / inverses:

```rust
use qip::*;

let n = 3;
let mut b = OpBuilder::new();
let ra = b.register(n) ?;
let rb = b.register(n) ?;

fn gamma(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> (Register, Register) {
    let (ra, rb) = b.cnot(ra, rb);
    let (rb, ra) = b.cnot(rb, ra);
    (ra, rb)
}

wrap_fn!(gamma_op, gamma, ra, rb);
invert_fn!(inv_gamma_op, gamma_op);

// This program is equivalent to the identity (U^-1 U = I).
let (ra, rb) = program!(&mut b, ra, rb;
    gamma_op ra, rb[2];
    inv_gamma_op ra, rb[2];
)?;
```

Functions in the `program!` macro may have a single argument, which is passed after the registers. This argument must be
included in the `wrap_fn!` call as well as the `invert_fn!` call.

```rust
use qip::*;

let mut b = OpBuilder::new();
let r = b.qubit();

fn rz(b: &mut dyn UnitaryBuilder, r: Register, theta: f64) -> Register {
    b.rz(r, theta)
}

wrap_fn!(rz_op(theta: f64), rz, r);
invert_fn!(inv_rz_op(theta: f64), rz_op);

let r = program!(&mut b, r;
    rz_op(3.141) r;
    inv_rz_op(3.141) r;
)?;
```

Generics can be used by substituting the usual angle brackets for square.

```rust
use qip::*;

let mut b = OpBuilder::new();
let r = b.qubit();

fn rz<T: Into<f64>>(b: &mut dyn UnitaryBuilder, r: Register, theta: T) -> Register {
    b.rz(r, theta.into())
}

wrap_fn!(rz_op[T: Into<f64>](theta: T), rz, r);
invert_fn!(inv_rz_op[T: Into<f64>](theta: T), rz_op);

let r = program!(&mut b, r;
    rz_op(3.141_f32) r;
    inv_rz_op(3.141_f32) r;
)?;
```

