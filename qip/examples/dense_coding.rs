use qip::prelude::*;
use std::num::NonZeroUsize;

fn run_alice<P: Precision, CB: CircuitBuilder>(b: &mut LocalBuilder<P>, epr_alice: CB::Register , bit_a: bool, bit_b: bool) {

    match (bit_a, bit_b) {
        (false, false) => epr_alice,
        (false, true) => b.x(epr_alice),
        (true, false) => b.z(epr_alice),
        (true, true) => b.y(epr_alice),

    }

}




fn main() -> Result<(), CircuitError> {

    let bits_a = vec![true, false, true, false];
    let bits_b = vec![true, true, false, false];

}