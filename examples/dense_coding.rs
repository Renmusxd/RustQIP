extern crate qip;

use qip::common_circuits::epr_pair;
use qip::*;

fn run_alice(b: &mut OpBuilder, epr_alice: Qubit, bit_a: bool, bit_b: bool) -> Qubit {
    match (bit_a, bit_b) {
        (false, false) => epr_alice,
        (false, true) => b.x(epr_alice),
        (true, false) => b.z(epr_alice),
        (true, true) => b.y(epr_alice),
    }
}

fn run_bob(b: &mut OpBuilder, q_alice: Qubit, epr_bob: Qubit) -> (bool, bool) {
    let (q_alice, q_bob) = condition(b, q_alice, epr_bob, |c, q| Ok(c.not(q))).unwrap();
    let q_alice = b.hadamard(q_alice);
    let q = b.merge(vec![q_bob, q_alice]);
    let (q, m) = b.measure(q);

    let (_, measurements) = run_local::<f64>(&q).unwrap();
    let (m, _) = measurements.get_measurement(&m).unwrap();

    ((m & 2) == 2, (m & 1) == 1)
}

fn main() -> Result<(), &'static str> {
    let bit_a = true;
    let bit_b = false;

    let bits_a = vec![true, false, true, false];
    let bits_b = vec![true, true, false, false];

    for (bit_a, bit_b) in bits_a.into_iter().zip(bits_b.into_iter()) {
        let mut b = OpBuilder::new();
        let (epr_alice, epr_bob) = epr_pair(&mut b, 1);
        let q_alice = run_alice(&mut b, epr_alice, bit_a, bit_b);
        let (bob_a, bob_b) = run_bob(&mut b, q_alice, epr_bob);

        println!(
            "Alice: ({:?},{:?})  \tBob: ({:?},{:?})",
            bit_a, bit_b, bob_a, bob_b
        );
    }

    Ok(())
}
