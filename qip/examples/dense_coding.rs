use qip::prelude::*;
use std::num::NonZeroUsize;
use qip::builder::Qudit;

/// Encode the two classical bits Alice wants to communicate to Bob.
///
/// Depending on the classical bits combination a different gate is applied:
/// 00: Do nothing (or apply Identity gate)
/// 01: Apply Pauli-X gate
/// 10: Apply Pauli-Z gate
/// 11: Apply Pauli-Y gate (or apply Pauli-Z gate followed by a Pauli-X gate)
///
/// Returns Alice qubit with applied gate.
///
/// https://en.wikipedia.org/wiki/Superdense_coding#Encoding
fn run_alice<P: Precision, CB: CliffordTBuilder<P>>(b: &mut CB, epr_alice: CB::Register , bit_a: bool, bit_b: bool) -> CB::Register {

    match (bit_a, bit_b) {
        (false, false) => epr_alice,
        (false, true) => b.x(epr_alice),
        (true, false) => b.z(epr_alice),
        (true, true) => b.y(epr_alice),

    }

}


/// Decode the message Alice transmitted to Bob.
///
/// Bob applies the restoration operation on his qubit and the one transmitted
/// by Alice to decode the original message. After restoration:
/// |00>: 00
/// |10>: 10
/// |01>: 01
/// |11>: 11
///
/// Returns a pair of classical bits.
///
/// https://en.wikipedia.org/wiki/Superdense_coding#Decoding
fn run_bob<P: Precision>(b: &mut LocalBuilder<P>, r_alice: Qudit, epr_bob: Qudit) -> (bool, bool) {
    let (r_alice, r_bob) = b.cnot(r_alice, epr_bob).unwrap();
    let r_alice = b.h(r_alice);
    let r = b.merge_two_registers(r_bob, r_alice);
    let (r, m) = b.measure(r);
    let (_, measurements) = b.calculate_state();
    let (m, _) = measurements.get_measurement(m);
    ((m & 2) == 2,  (m & 1) == 1)
    
}


fn main() {

    let n = NonZeroUsize::new(1).unwrap();
    let bits_a = vec![true, false, true, false];
    let bits_b = vec![true, true, false, false];

    for (bit_a, bit_b) in bits_a.into_iter().zip(bits_b.into_iter()) {
        let mut b = LocalBuilder::<f64>::default();
        let epr_alice = b.register(n);
        let epr_bob = b.register(n);

        let r_alice = run_alice(&mut b, epr_alice, bit_a, bit_b);
        let (bob_a, bob_b) = run_bob(&mut b, r_alice, epr_bob);

        println!(
            "Alice: ({:?},{:?}) \tBob: ({:?}, {:?})",
            bit_a, bit_b, bob_a, bob_b
        );

    }

}