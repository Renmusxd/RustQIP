extern crate qip;

use qip::common_circuits::epr_pair;
use qip::errors::CircuitError;
use qip::*;

fn run_alice(b: &mut OpBuilder, epr_alice: Register, bit_a: bool, bit_b: bool) -> Register {
    match (bit_a, bit_b) {
        (false, false) => epr_alice,
        (false, true) => b.x(epr_alice),
        (true, false) => b.z(epr_alice),
        (true, true) => b.y(epr_alice),
    }
}

fn run_bob(b: &mut OpBuilder, r_alice: Register, epr_bob: Register) -> (bool, bool) {
    let (r_alice, r_bob) = b.cnot(r_alice, epr_bob);
    let r_alice = b.hadamard(r_alice);
    let r = b.merge(vec![r_bob, r_alice]).unwrap();
    let (r, m) = b.measure(r);

    let (_, measurements) = run_local::<f64>(&r).unwrap();
    let (m, _) = measurements.get_measurement(&m).unwrap();

    ((m & 2) == 2, (m & 1) == 1)
}

fn main() -> Result<(), CircuitError> {
    let bit_a = true;
    let bit_b = false;

    let bits_a = vec![true, false, true, false];
    let bits_b = vec![true, true, false, false];

    for (bit_a, bit_b) in bits_a.into_iter().zip(bits_b.into_iter()) {
        let mut b = OpBuilder::new();
        let (epr_alice, epr_bob) = epr_pair(&mut b, 1);
        let r_alice = run_alice(&mut b, epr_alice, bit_a, bit_b);
        let (bob_a, bob_b) = run_bob(&mut b, r_alice, epr_bob);

        println!(
            "Alice: ({:?},{:?})  \tBob: ({:?},{:?})",
            bit_a, bit_b, bob_a, bob_b
        );
    }

    Ok(())
}
