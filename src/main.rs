extern crate qip;

use qip::pipeline::run;
use qip::qubits::*;

fn main() {
    let mut builder = qip::qubits::OpBuilder::new();

    let q1 = builder.qubit(1);
    let q2 = builder.qubit(13);
    let q3 = builder.qubit(13);

    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);

    let (q2, q3) = c.swap(q2, q3);

    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    println!("Qs: {:?}, {:?}, {:?}", q1, q2, q3);

    let q4 = builder.merge(vec![q1, q2, q3]);

    println!("Qs: {:?}", q4);

    let out = run(&q4);
}
