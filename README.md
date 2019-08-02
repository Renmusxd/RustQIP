# RustQIP

Quantum Computing library leveraging graph building to build efficient quantum circuit
simulations.

See all the examples in the [examples directory](https://github.com/Renmusxd/RustQIP/tree/master/examples) of the Github repository.

# Example (CSWAP)
Here's an example of a small circuit where two groups of qubits are swapped conditioned on a
third. This circuit is very small, only three operations plus a measurement, so the boilerplate
can look quite large in compairison, but that setup provides the ability to construct circuits
easily and safely when they do get larger.
```rust
use qip::*;

// Make a new circuit builder.
let mut b = OpBuilder::new();

// Make three logical groups of qubits of sizes 1, 3, 3 (7 qubits total).
let q = b.qubit(1)?;
let qa = b.qubit(3)?;
let qb = b.qubit(3)?;

// We will want to feed in some inputs later, hang on to the handles
// so we don't need to actually remember any indices.
let a_handle = qa.handle();
let b_handle = qb.handle();

// Define circuit
// First apply an H to q1
let q = b.hadamard(q);
// Then run this subcircuit conditioned on q1, applied to q2 and q3
let (q, _) = condition(&mut b, q, (qa, qb), |c, (qa, qb)| {
    c.swap(qa, qb)
})?;
// Finally apply H to q1 again.
let q = b.hadamard(q);

// Add a measurement to the first qubit, save a reference so we can get the result later.
let (q, m_handle) = b.measure(q);

// Now q is the end result of the above circuit, and we can run the circuit by referencing it.

// Make an initial state: |0,000,001>
let initial_state = [a_handle.make_init_from_index(0)?,
                     b_handle.make_init_from_index(1)?];
// Run circuit with a given precision.
let (_, measured) = run_local_with_init::<f64>(&q, &initial_state)?;

// Lookup the result of the measurement we performed using the handle.
let (result, p) = measured.get_measurement(&m_handle).unwrap();

// Print the measured result
println!("Measured: {:?} (with chance {:?})", result, p);
```
