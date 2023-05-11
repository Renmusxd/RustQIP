#[cfg(feature = "macros")]
use qip::prelude::*;
#[cfg(feature = "macros")]
use qip_macros::program;

#[cfg(feature = "macros")]
fn gamma<P: Precision, CB: CliffordTBuilder<P>>(
    cb: &mut CB,
    ra: CB::Register,
) -> Result<CB::Register, CircuitError> {
    let ra = cb.not(ra);
    Ok(ra)
}

#[cfg(feature = "macros")]
fn main() -> Result<(), CircuitError> {
    let mut b = LocalBuilder::<f64>::default();

    let ra = b.qubit();
    let rb = b.qubit();

    let ra = b.h(ra);
    let (ra, rb) = program!(&mut b; ra, rb;
        control gamma ra, rb;
    )?;

    let r = b.merge_two_registers(ra, rb);
    let (_, handle) = b.measure_stochastic(r);

    let (state, measures) = b.calculate_state();
    println!("{:?}", state);
    println!("{:?}", measures.get_stochastic_measurement(handle));
    Ok(())
}

#[cfg(not(feature = "macros"))]
fn main() {
    panic!("Macros not enabled.")
}
