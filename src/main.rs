extern crate qip;

use qip::pipeline::run;
use qip::qubits::*;

fn main() {
    let mut builder = qip::qubits::OpBuilder::new();

    let q1 = builder.qubit(1);
    let q2 = builder.qubit(5);
    let q3 = builder.qubit(5);

    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);
    c.swap(q2, q3); // we don't need the output from here
    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    let (q1, handle) = builder.measure(q1);

    let (out, results) = run(&q1);

    println!("Results: {:?}", results.results[&handle]);
}
