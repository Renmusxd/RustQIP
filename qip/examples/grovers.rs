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

    //run_local(&r).map(|(s, _)| s) ???
    let (_, handle) = b.measure_stochastic(r);
    Ok(())
}

fn main() {}
