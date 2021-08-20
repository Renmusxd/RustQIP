use qip::*;

fn circuit1() -> Vec<f64> {
    // build 2 qubits
    let mut builder = OpBuilder::new();
    let qubit1 = builder.qubit();
    let qubit2 = builder.qubit();

    // transform qubits from |0> to |1>
    let qubit1 = builder.not(qubit1);
    let qubit2 = builder.not(qubit2);

    // apply hadamard and cnot
    let qubit1 = builder.hadamard(qubit1);
    let (qubit1, qubit2) = builder.cnot(qubit1, qubit2);

    // apply rz to qubit2
    let qubit2 = builder.rz(qubit2, std::f64::consts::FRAC_PI_3);

    // apply hadamard to both qubits
    let qubit1 = builder.hadamard(qubit1);
    let qubit2 = builder.hadamard(qubit2);

    // merge both qubits
    let merged = builder.merge(vec![qubit1, qubit2]).unwrap();

    // add a measurement of all possible states
    let (merged, m_handle) = builder.stochastic_measure(merged);

    // run and get probabilities
    let (_, mut measured) = run_local::<f64>(&merged).unwrap();
    let probabilities = measured.pop_stochastic_measurements(m_handle).unwrap();

    probabilities
}

fn circuit2() -> Vec<f64> {
    // build 2 qubits
    let mut builder = OpBuilder::new();
    let qubit1 = builder.qubit();
    let qubit2 = builder.qubit();

    // transform qubits from |0> to |1>
    let qubit1 = builder.not(qubit1);
    let qubit2 = builder.not(qubit2);

    // apply hadamard and cnot
    let qubit1 = builder.hadamard(qubit1);
    let (qubit1, qubit2) = builder.cnot(qubit1, qubit2);

    // apply rz to qubit2
    let qubit2 = builder.rz(qubit2, 2.0 * std::f64::consts::FRAC_PI_3);

    // apply hadamard to both qubits
    let qubit1 = builder.hadamard(qubit1);
    let qubit2 = builder.hadamard(qubit2);

    // merge both qubits
    let merged = builder.merge(vec![qubit1, qubit2]).unwrap();

    // add a measurement of all possible states
    let (merged, m_handle) = builder.stochastic_measure(merged);

    // run and get probabilities
    let (_, mut measured) = run_local::<f64>(&merged).unwrap();
    let probabilities = measured.pop_stochastic_measurements(m_handle).unwrap();

    probabilities
}

fn circuit3() -> Vec<f64> {
    // build 2 qubits
    let mut builder = OpBuilder::new();
    let qubit1 = builder.qubit();
    let qubit2 = builder.qubit();

    // transform qubits from |0> to |1>
    let qubit1 = builder.not(qubit1);
    let qubit2 = builder.not(qubit2);

    // apply hadamard and cnot
    let qubit1 = builder.hadamard(qubit1);
    let (qubit1, qubit2) = builder.cnot(qubit1, qubit2);

    // apply rz to both qubits
    let qubit1 = builder.rz(qubit1, std::f64::consts::FRAC_PI_3);
    let qubit2 = builder.rz(qubit2, 2.0 * std::f64::consts::FRAC_PI_3);

    // apply hadamard to both qubits
    let qubit1 = builder.hadamard(qubit1);
    let qubit2 = builder.hadamard(qubit2);

    // merge both qubits
    let merged = builder.merge(vec![qubit1, qubit2]).unwrap();

    // add a measurement of all possible states
    let (merged, m_handle) = builder.stochastic_measure(merged);

    // run and get probabilities
    let (_, mut measured) = run_local::<f64>(&merged).unwrap();
    let probabilities = measured.pop_stochastic_measurements(m_handle).unwrap();

    probabilities
}

/// Bell inequality:
/// |P(a, b) - P(a, c)| - P(b, c) <= 1
/// To get P(a, b), P(a, c) and P(b, c) we use the 3 circuits presented in:
/// https://arxiv.org/pdf/1712.05642.pdf - II.IMPLEMENTED EXPERIMENTS -> 3. Bell's inequality
///
/// A violation of the inequality is expected in any quantum computer or simulator.
fn main() -> Result<(), CircuitError> {
    println!("Bell inequality: |P(a, b) - P(a, c)| - P(b, c) <= 1");

    let a_b = circuit1();
    let p_of_a_b = (a_b[0] + a_b[3]) - (a_b[1] + a_b[2]);
    println!("P(a, b) = {:.2}", p_of_a_b);

    let a_c = circuit2();
    let p_of_a_c = (a_c[0] + a_c[3]) - (a_c[1] + a_c[2]);
    println!("P(a, c) = {:.2}", p_of_a_c);

    let b_c = circuit3();
    let p_of_b_c = (b_c[0] + b_c[3]) - (b_c[1] + b_c[2]);
    println!("P(b, c) = {:.2}", p_of_b_c);

    let left_side = num::abs(p_of_a_b - p_of_a_c) - p_of_b_c;
    println!(
        "|{:.2} - {:.2}| - ({:.2}) = {:.2} IS NOT <= 1",
        p_of_a_b, p_of_a_c, p_of_b_c, left_side
    );

    assert!(left_side > 1.0);

    Ok(())
}
