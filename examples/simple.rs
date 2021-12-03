use qip::prelude::*;

fn gamma<P: Precision, CB: CliffordTBuilder<P>>(
    cb: &mut CB,
    mut rs: Vec<CB::Register>,
) -> Result<Vec<CB::Register>, CircuitError> {
    let r = rs.pop().unwrap();
    let r = cb.not(r);
    rs.push(r);
    Ok(rs)
}

fn main() {
    let mut cb = LocalBuilder::<f64>::default();

    let ra = cb.qubit();
    let rb = cb.qubit();

    let ra = cb.h(ra);
    let (ra, rb) = program!(&mut cb, ra, rb;
        control gamma ra, rb;
    )
    .unwrap();

    let r = cb.merge_two_registers(ra, rb);
    let (_, handle) = cb.measure_stochastic(r);

    let (state, measures) = cb.calculate_state();
    println!("{:?}", state);
    println!("{:?}", measures.get_stochastic_measurement(handle));
}
