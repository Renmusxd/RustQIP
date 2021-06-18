use qip::*;

fn main() -> Result<(), CircuitError> {
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
    let initial_state = [ha.make_init_from_index(0)?, hb.make_init_from_index(1)?];

    // Run circuit
    let (_, measured) = run_local_with_init::<f64>(&q, &initial_state)?;

    println!("{:?}", measured.get_measurement(&m1));

    Ok(())
}
