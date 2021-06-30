use qip::common_circuits::epr_pair;
use qip::*;

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
fn run_alice(b: &mut OpBuilder, epr_alice: Register, bit_a: bool, bit_b: bool) -> Register {
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
fn run_bob(b: &mut OpBuilder, r_alice: Register, epr_bob: Register) -> (bool, bool) {
    let (r_alice, r_bob) = b.cnot(r_alice, epr_bob);
    let r_alice = b.hadamard(r_alice);
    let r = b.merge(vec![r_bob, r_alice]).unwrap();
    let (r, m) = b.measure(r);

    let (_, measurements) = run_local::<f64>(&r).unwrap();
    let (m, _) = measurements.get_measurement(&m).unwrap();

    ((m & 2) == 2, (m & 1) == 1)
}

/// Superdense coding example: Packing two classical bits into one qubit.
///
/// - A third party (Charlie) entangles two qubits, sending one to Alice and the other to Bob.
/// - Alice wants to send a message to Bob, she encodes the entangled qubit she owns and sends it to Bob.
/// - Bob now has the two entangled qubits, after decoding he can read the message Alice sent.
/// - Alice makes no measurement but Bob does, destroying the entanglement at that time.
///
/// https://en.wikipedia.org/wiki/Superdense_coding
fn main() -> Result<(), CircuitError> {
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
