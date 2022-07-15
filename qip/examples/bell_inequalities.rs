use qip::prelude::*;
use std::num::NonZeroUsize;

fn circuit1() -> Vec<f64> {

    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(2).unwrap();

    let q = b.qubit();
    let ra = b.register(n);
    let r = program!() //Not sure what to use here
}