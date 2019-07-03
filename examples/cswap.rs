extern crate num;
extern crate qip;

use qip::*;

fn main() -> Result<(), &'static str> {
    // Setup inputs
    let mut b = OpBuilder::new();
    let q1 = b.qubit(1)?;
    let q2 = b.qubit(3)?;
    let q3 = b.qubit(3)?;

    // We will want to feed in some inputs later.
    let h2 = q2.handle();
    let h3 = q3.handle();

    // Define circuit
    let q1 = b.hadamard(q1);

    let mut c = b.with_context(q1);
    let _ = c.swap(q2, q3)?;
    let q1 = c.release_qubit();

    let q1 = b.hadamard(q1);

    let (q1, m1) = b.measure(q1);

    // Print circuit diagram
    qip::run_debug(&q1);

    // Run circuit
    let (_, measured) = run_local_with_init::<f64>(&q1, &[
        h2.make_init_from_index(0)?,
        h3.make_init_from_index(1)?,
    ]);

    println!("{:?}", measured.get_measurement(m1));

    Ok(())
}