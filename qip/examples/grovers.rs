use qip::builder::Qudit;
use qip::builder_traits::UnitaryBuilder;
use qip::prelude::*;
use std::num::NonZeroUsize;
//-> Result<(), CircuitError>

fn prepare_state<P: Precision>(n: u64) -> Result<(), CircuitError> {
    let mut b = LocalBuilder::<f64>::default();

    let n = NonZeroUsize::new(n as usize).unwrap();
    let r = b.register(n);
    let r = b.h(r);

    let anc = b.qubit();
    let anc = b.not(anc);
    let anc = b.h(anc);

    let r = b.merge_two_registers(r, anc);

    let (_, handle) = b.measure_stochastic(r);

    let (state, measures) = b.calculate_state();
    println!("{:?}", state);
    println!("{:?}", measures.get_stochastic_measurement(handle));
    Ok(())
}
#[cfg(feature = "macros")]
fn apply_us<P: Precision>(
    b: &mut dyn UnitaryBuilder<P>,
    search: Qudit,
    ancillary: Qudit,
    x0: u64,
) -> Result<(Qudit, Qudit), CircuitError> {
    let search = b.h(search);
    let (search, ancillary) = program!(b, search, ancillary, |x| {
        (0, if x == 0 { std::f64::consts::PI } else { 0.0 })
    })?;
    let search = b.h(search);
    Ok((search, ancillary))
}

#[cfg(feature = "macros")]
fn apply_uw(
    b: &mut dyn UnitaryBuilder,
    search: Qudit,
    ancillary: Qudit,
    x0: u64,
) -> Result<(Qudit, Qudit), CircuitError> {
    // Need to move the x0 value into the closure.
    program!(b, search, ancillary, move |x| ((x == x0) as u64, 0.0))
}


// #[cfg(feature = "macros")]
// fn apply_grover_iteration<P: Precision>(
//     x: u64,
//     s: LocalQuantumState<P>,
// ) -> Result<LocalQuantumState<P>, CircuitError> {
//     let mut b = OpBuilder::new();
//     let r = b.register(s.n() - 1)?;
//     let anc = b.qubit();

//     let (r, anc) = apply_uw(&mut b, r, anc, x)?;
//     let (r, _) = apply_us(&mut b, r, anc)?;
//     run_with_state(&r, s).map(|(s, _)| s)
// }


fn main() {}
