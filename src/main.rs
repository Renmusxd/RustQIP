extern crate qip;

use qip::qubits::*;
use qip::pipeline::{run_local, run_local_with_init};
use qip::pipeline_debug::run_debug;
use qip::qfft;
use qip::state_ops::from_reals;

fn main() {
    let mut builder = qip::qubits::OpBuilder::new();
//
//    let (q1, q1_handle) = builder.qubit_and_handle(1);
//    let (q2, q2_handle) = builder.qubit_and_handle(2);
//    let (q3, q3_handle) = builder.qubit_and_handle(2);
//
//    let q1 = builder.hadamard(q1);
//
//    let mut c = builder.with_context(q1);
//    let (q2, q3) = c.swap(q2, q3); // we don't need the output from here
//    let q1 = c.release_qubit();
//
//    let q1 = builder.hadamard(q1);
//
//    let (q1, handle) = builder.measure(q1);
//
//    run_debug(&q1);

//    let (out, results) = run_local_with_init(&q1, &[
//        q1_handle.make_init_from_index(0).unwrap(),
//        q2_handle.make_init_from_index(1).unwrap(),
//        q3_handle.make_init_from_index(2).unwrap(),
//    ]);
//
//    for (i, c) in out.state.iter().enumerate() {
//        if c.norm_sqr() > 0.0 {
//            println!("|{:05b}>\t{:.*}", i, 3, out.state[i].norm_sqr());
//        }
//    }
//
//    println!("Results: {:?}", results.results[&handle]);

    let n = 10;
    let (q, h) = builder.qubit_and_handle(n);

    let q = qfft::qfft(&mut builder, q);

//    run_debug(&q);
    let state: Vec<f64> = (0 .. 1 << n).map(|i| (i as f64 * 0.005 * std::f64::consts::PI).cos()).collect();
    let mag: f64 = state.iter().map(|f| f.powi(2)).sum();
    let state: Vec<f64> = state.iter().map(|f| f / mag.sqrt()).collect();

    let state = from_reals(state.as_slice());
    let copy_state = state.clone();

    let (out, results) = run_local_with_init(&q, &[
        h.make_init_from_state(state).unwrap()
    ]);
    for (i, c) in out.state.iter().enumerate() {
        if c.norm_sqr() > 0.0 {
            println!("{:?}\t{:.*}\t{:.*}", i, 3, copy_state[i].norm_sqr(), 3, out.state[i].norm_sqr());
        }
    }
}