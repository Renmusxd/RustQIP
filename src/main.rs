extern crate qip;

extern crate num;

use num::Complex;

use qip::qubits::*;
use qip::pipeline;
use qip::state_ops;
use qip::pipeline_debug::run_debug;
use qip::qfft;
use qip::state_ops::from_reals;
use qip::utils::flip_bits;

fn main() {
    let mut builder = qip::qubits::OpBuilder::new();
    let (q1, h1) = builder.qubit_and_handle(1);
    let (q2, h2) = builder.qubit_and_handle(1);
    let (q3, h3) = builder.qubit_and_handle(1);
    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);
    let (q2, q3) = c.swap(q2, q3);
    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    let (q1, m1) = builder.measure(q1);

    run_debug(&q1);

    let (out, measured) = pipeline::run_local_with_init(&q1, &[
        h2.make_init_from_index(0).unwrap(),
        h3.make_init_from_index(1).unwrap(),
    ]);

    println!("{:?}", measured.results.get(&m1));

    for i in 0 .. out.state.len() {
        println!("|{:?}>: {:.*}\t{:.*}i", i, 3, out.state[i].re, 3, out.state[i].im);
    }
}