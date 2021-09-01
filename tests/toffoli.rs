use qip::*;

// https://en.wikipedia.org/wiki/Quantum_logic_gate#Toffoli_(CCNOT)_gate
type Entry = (u64, u64, u64);
const fn truth_table() -> [(Entry, Entry); 8] {
    [
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 0)),
    ]
}

#[test]
fn test_toffoli() -> Result<(), CircuitError> {
    for (input, output) in truth_table() {
        // create a builder
        let mut b = OpBuilder::new();

        // all qubits are initially in the 0 state
        let mut q1 = b.qubit();
        let mut q2 = b.qubit();
        let mut q3 = b.qubit();

        // change inputs as truth table
        if input.0 == 1 {
            q1 = b.not(q1);
        }
        if input.1 == 1 {
            q2 = b.not(q2);
        }
        if input.2 == 1 {
            q3 = b.not(q3);
        }

        // apply the Toffoli gate
        let q = b.ccnot(q1, q2, q3);

        // add measurements to all the qubits
        let (q1, handle1) = b.measure(q.0);
        let (q2, handle2) = b.measure(q.1);
        let (q3, handle3) = b.measure(q.2);

        // run each qubit
        let (_, measured1) = run_local::<f64>(&q1).unwrap();
        let m1 = measured1.get_measurement(&handle1).unwrap();

        let (_, measured2) = run_local::<f64>(&q2).unwrap();
        let m2 = measured2.get_measurement(&handle2).unwrap();

        let (_, measured3) = run_local::<f64>(&q3).unwrap();
        let m3 = measured3.get_measurement(&handle3).unwrap();

        // compare results with truth table outputs
        assert_eq!(m1.0, output.0);
        assert_eq!(m2.0, output.1);
        assert_eq!(m3.0, output.2);
    }

    Ok(())
}
