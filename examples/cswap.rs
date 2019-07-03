extern crate num;
extern crate qip;

use qip::*;

fn main() {
    let sn = 5;
    let mut builder = OpBuilder::new();
    let q1 = builder.qubit(1).unwrap();
    let (q2, h2) = builder.qubit_and_handle(sn).unwrap();
    let (q3, h3) = builder.qubit_and_handle(sn).unwrap();
    let q1 = builder.hadamard(q1);

    let mut c = builder.with_context(q1);
    let _ = c.swap(q2, q3).unwrap();
    let q1 = c.release_qubit();

    let q1 = builder.hadamard(q1);

    let (q1, m1) = builder.measure(q1);

    qip::run_debug(&q1);

    let (_, measured) = run_local_with_init::<f64>(&q1, &[
        h2.make_init_from_index(0).unwrap(),
        h3.make_init_from_index(1).unwrap(),
    ]);

    println!("{:?}", measured.get_measurement(m1));
}