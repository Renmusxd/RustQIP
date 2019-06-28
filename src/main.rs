extern crate num;
extern crate qip;

use qip::pipeline::{make_circuit_matrix, QuantumState, run_local_with_init};
use qip::pipeline_debug::run_debug;
use qip::qubits::*;
use qip::state_ops::{from_reals, make_op_matrix};
use qip::utils::flip_bits;

fn main() {
    let sn = 4;
    let mut builder = OpBuilder::<f32>::new();
    let (q1, h1) = builder.qubit_and_handle(1).unwrap();
    let (q2, h2) = builder.qubit_and_handle(sn).unwrap();
    let (q3, h3) = builder.qubit_and_handle(sn).unwrap();
    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);
    let (q2, q3) = c.swap(q2, q3).unwrap();
    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    let (q1, m1) = builder.measure(q1);

    run_debug(&q1);

    let (out, measured) = run_local_with_init(&q1, &[
        h2.make_init_from_index(0).unwrap(),
        h3.make_init_from_index(1).unwrap(),
    ]);

    println!("{:?}", measured.results.get(&m1));
}