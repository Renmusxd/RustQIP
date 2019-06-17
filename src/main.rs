extern crate qip;

use qip::qubits::*;
use qip::pipeline::*;
use qip::multi_ops::make_ops_matrix;
use qip::state_ops::*;

fn main() {
    let mut builder = qip::qubits::OpBuilder::new();

    let (q1, q1_handle) = builder.qubit_and_handle(1);
    let (q2, q2_handle) = builder.qubit_and_handle(2);
    let (q3, q3_handle) = builder.qubit_and_handle(2);

    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);
    let (q2, q3) = c.swap(q2, q3); // we don't need the output from here
    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    let (q1, handle) = builder.measure(q1);

    // TODO clean up this initialization API
    let (out, results) = run_with_state(&q1, |qs| {
        let n: u64 = qs.iter().map(|q| q.indices.len() as u64).sum();
        LocalQuantumState::new_from_initial_states(n, vec![
            (q1_handle, InitialState::Index(0)),
            (q2_handle, InitialState::Index(1)),
            (q3_handle, InitialState::Index(2)),
        ])
    });

    for (i, c) in out.state.iter().enumerate() {
        if c.norm_sqr() > 0.0 {
            println!("|{:05b}>\t{:.*}", i, 3, out.state[i].norm_sqr());
        }
    }

    println!("Results: {:?}", results.results[&handle]);
}