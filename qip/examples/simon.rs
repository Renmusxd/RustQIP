use bit_vec::BitVec;
use qip::prelude::*;
use std::collections::HashSet;

/// Simon's algorithm demo implementation.
///
/// https://en.wikipedia.org/wiki/Simon's_problem
/// https://qiskit.org/textbook/ch-algorithms/simon.html
fn main() -> CircuitResult<()> {
    println!("2-bit secrets:");
    println!();

    // trivial case: secret = "00"
    let secret = BitVec::from_elem(2, false);

    let (_measurement, likelihood) = simon_circuit(secret.clone())?;
    let likelihood: f64 = format!("{:.2}", likelihood).parse().unwrap();
    if (likelihood - 1.0 / (secret.len() as f64).exp2()).abs() < 0.01 {
        println!(
            "Secret string is {} as this is uniformly distributed with prob 1/(2.exp(n))",
            format!("{:#04b}", 0)
        );
    }

    // non-trivial cases: secret = "11" or "10" or "01"
    let secret1 = BitVec::from_elem(2, true);
    let mut secret2 = BitVec::from_elem(2, true);
    secret2.set(1, false);
    let mut secret3 = BitVec::from_elem(2, true);
    secret3.set(0, false);
    let secrets = vec![secret1, secret2, secret3];

    for secret in secrets {
        loop {
            // run as many times until we get a non zero answer
            let (measurement, likelihood) = simon_circuit(secret.clone())?;
            if measurement != 0 {
                let likelihood: f64 = format!("{:.2}", likelihood).parse().unwrap();
                if (likelihood - 1.0 / ((secret.len() as f64) - 1.0).exp2()).abs() < 0.01 {
                    println!(
                        "Secret string is {} as the likelhood is 1/2.exp(n-1)",
                        format!("{:#04b}", measurement)
                    );
                }
                break;
            }
        }
    }

    println!();
    println!("3-bit secrets:");
    println!();

    // trivial case: secret = "000"
    let secret = BitVec::from_elem(3, false);

    let (_measurement, likelihood) = simon_circuit(secret.clone())?;
    let likelihood: f64 = format!("{:.3}", likelihood).parse().unwrap();
    if (likelihood - 1.0 / (secret.len() as f64).exp2()).abs() < 0.01 {
        println!(
            "Secret string is {} as this is uniformly distributed with prob 1/(2.exp(n))",
            format!("{:#05b}", 0)
        );
        println!();
    }

    // non-trivial cases: secret = "001" or "010" or "011" or "100" or "101" or "110" or "111"
    let mut secret1 = BitVec::from_elem(3, false);
    secret1.set(2, true);
    let mut secret2 = BitVec::from_elem(3, false);
    secret2.set(1, true);
    let mut secret3 = BitVec::from_elem(3, true);
    secret3.set(0, false);
    let mut secret4 = BitVec::from_elem(3, false);
    secret4.set(0, true);
    let mut secret5 = BitVec::from_elem(3, true);
    secret5.set(1, false);
    let mut secret6 = BitVec::from_elem(3, true);
    secret6.set(2, false);
    let secret7 = BitVec::from_elem(3, true);

    let secrets = vec![
        secret1, secret2, secret3, secret4, secret5, secret6, secret7,
    ];

    for secret in secrets {
        let mut seen = HashSet::new();

        loop {
            // run as many times until we get secret.len() different outputs
            // (drop trivial 000 results).
            let (measurement, _likelihood) = simon_circuit(secret.clone())?;
            if measurement != 0 {
                seen.insert(measurement);

                if seen.len() == secret.len() {
                    break;
                }
            }
        }

        // confirm known secret string with results
        for measured in seen {
            // format and reverse
            let measured = format!("{:#05b}", measured);
            let measured = &measured[2..measured.len()];
            let measured: String = measured.chars().rev().collect();

            // all dot products should be zero
            assert_eq!((dot_product(format!("{:?}", secret), &measured)), 0);
            println!(
                "Secret: 0b{:?}, Measured: 0b{} - Confirmation: 0b{:?}.0b{} = {} (mod 2)",
                secret,
                &measured,
                secret,
                &measured,
                dot_product(format!("{:?}", secret), &measured)
            );
        }

        // TODO: solve system of equations by gaussian elimination
        println!();
    }

    Ok(())
}

/// Create and run a Simon's circuit.
/// Return measurement and likelihood for each round.
fn simon_circuit(secret: BitVec) -> CircuitResult<(usize, f64)> {
    let mut b = LocalBuilder::<f64>::default();
    let n = secret.len();

    // create registers for string length
    let input_register = b.qudit(n).unwrap();
    let output_register = b.qudit(n).unwrap();

    // apply the first hadamard
    let input_register = b.h(input_register);

    // apply the oracle
    let (input_register, output_register) =
        simon_oracle(&mut b, input_register, output_register, secret)?;

    // meassure the second register but drop the results
    let (r, m) = b.measure(output_register);
    let (_, measurements) = b.calculate_state();
    let (_, _) = measurements.get_measurement(m);

    // apply second hadamard
    let input_register = b.h(input_register);

    // meassure the first register and return the measurement
    let (r, m) = b.measure(input_register);
    let (_, measurements) = b.calculate_state();
    Ok(measurements.get_measurement(m))
}

/// Apply the Simon's oracle.
///
/// Some references from other implementations:
/// - https://github.com/qiskit-community/qiskit-textbook/blob/589c64d66c8743c123c9704d9b66cda4d476dbff/qiskit-textbook-src/qiskit_textbook/tools/__init__.py#L26
/// - https://quantumcomputing.stackexchange.com/questions/15567/in-simons-algorithm-is-there-a-general-method-to-define-an-oracle-given-a-cert
fn simon_oracle<P, CB>(
    b: &mut CB,
    input_register: CB::Register,
    output_register: CB::Register,
    secret: BitVec,
) -> CircuitResult<(CB::Register, CB::Register)>
    where P: Precision, CB: CliffordTBuilder<P> {
    // length of the secret string
    let n = secret.len();

    // split the registers in qubits
    let mut input_qubits = b.split_all_register(input_register);
    let mut output_qubits = b.split_all_register(output_register);

    // copy input qubits to output qubits
    for i in 0..n {
        let (qi, qo) = b.cnot(input_qubits.remove(i), output_qubits.remove(i))?;
        input_qubits.insert(i, qi);
        output_qubits.insert(i, qo);
    }

    // get the index of the first "1" found in the secret if any.
    if secret.any() {
        let i = secret
            .iter()
            .enumerate()
            .find_map(|pair| if pair.1 { Some(pair.0) } else { None })
            .unwrap();

        // add significant bit if secret string is 1 at position
        for q in 0..n {
            if secret.get(q).unwrap() {
                let (qi, qo) = b.cnot(input_qubits.remove(i), output_qubits.remove(q))?;
                input_qubits.insert(i, qi);
                output_qubits.insert(q, qo);
            }
        }
    }

    // merge back the individual qubits in registers
    let input_register = b.merge_registers(input_qubits).unwrap();
    let output_register = b.merge_registers(output_qubits).unwrap();

    Ok((input_register, output_register))
}

/// Do the dot prduct between the secret string and a result measurement.
fn dot_product(secret: String, result: &str) -> i32 {
    let mut accum = 0;
    for i in 0..secret.len() {
        accum += secret
            .chars()
            .nth(i)
            .unwrap()
            .to_string()
            .parse::<i32>()
            .unwrap()
            * result
            .chars()
            .nth(i)
            .unwrap()
            .to_string()
            .parse::<i32>()
            .unwrap();
    }
    accum % 2
}
