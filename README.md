# RustQIP

Quantum Computing library leveraging graph building to build efficient quantum circuit
simulations.

[![qip on crates.io](https://img.shields.io/crates/v/qip.svg)](https://crates.io/crates/qip)
[![qip docs](https://img.shields.io/badge/docs-docs.rs-orange.svg)](https://docs.rs/qip)

See all the examples in the [examples directory](https://github.com/Renmusxd/RustQIP/tree/master/examples) of the Github repository.

# Example (CSWAP)
Here's an example of a small circuit where two groups of qubits are swapped conditioned on a
third. This circuit is very small, only three operations plus a measurement, so the boilerplate
can look quite large in compairison, but that setup provides the ability to construct circuits
easily and safely when they do get larger.
```rust
use qip::*;

// Setup inputs
let mut b = OpBuilder::new();
let q = b.qubit();
let ra = b.register(3)?;
let rb = b.register(3)?;

// We will want to feed in some inputs later.
let ha = ra.handle();
let hb = rb.handle();

// Define circuit
let q = b.hadamard(q);

let (q, _, _) = b.cswap(q, ra, rb)?;
let q = b.hadamard(q);

let (q, m1) = b.measure(q);

// Print circuit diagram
qip::run_debug(&q)?;

// Initialize ra to |0> and rb to |1> using their handles.
// Make an initial state: |0,000,001>
let initial_state = [ha.make_init_from_index(0)?,
                     hb.make_init_from_index(1)?];

// Run circuit
let (_, measured) = run_local_with_init::<f64>(&q, &initial_state)?;

println!("{:?}", measured.get_measurement(&m1));
```
