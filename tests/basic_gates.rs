extern crate num;
extern crate qip;

use crate::pipeline::Representation;
use qip::{state_ops::from_reals, *};

// TODO: remove this if #23 lands
fn assert_almost_eq(a: f64, b: f64, prec: i32) {
    let mult = 10.0f64.powi(prec);
    let (a, b) = (a * mult, b * mult);
    let (a, b) = (a.round(), b.round());
    assert_eq!(a / mult, b / mult);
}

/// The gates we will be testing
enum Gate {
    PauliX,
    PauliY,
    PauliZ,
    Phase(f64),
    Hadamard,
}

/// Collect results after a gate runs
struct GateResult {
    output_state: Vec<num::Complex<f64>>,
    measurement: u64,
    likelihood: f64,
}

/// Run a qubit with provided initial state and gate
fn run(input_state: &Vec<num::Complex<f64>>, gate: Option<Gate>) -> GateResult {
    let mut b = OpBuilder::new();
    let q = b.qubit();
    let h = q.handle();

    // create initial state
    let initial_state = [h
        .make_init_from_state(vec![input_state[0], input_state[1]])
        .unwrap()];

    let q = match gate {
        Some(gate) => match gate {
            Gate::PauliX => b.x(q),
            Gate::PauliY => b.y(q),
            Gate::PauliZ => b.z(q),
            Gate::Phase(theta) => b.phase(q, theta),
            Gate::Hadamard => b.hadamard(q),
        },
        None => q,
    };

    let (q, m) = b.measure(q);
    let (state, measured) = run_local_with_init::<f64>(&q, &initial_state).ok().unwrap();
    let m = measured.get_measurement(&m).unwrap();

    let output_state = state.into_state(Representation::BigEndian);

    GateResult {
        output_state,
        measurement: m.0,
        likelihood: m.1,
    }
}

/// Prepare basis states |0> and |1> and just measure.
#[test]
fn no_gates() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    let result = run(&input_state, None);

    assert_eq!(result.measurement, 0);
    assert_eq!(result.likelihood, 1.0);
    assert_eq!(result.output_state, input_state);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    let result = run(&input_state, None);

    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);
    assert_eq!(result.output_state, input_state);

    Ok(())
}

/// Prepare basis states |0> and |1> and:
/// - apply pauli-x gate [Wikipedia: Pauli Gates]
/// - measure
///
/// [Wikipedia: Pauli Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z))
#[test]
fn pauli_x() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    let result = run(&input_state, Some(Gate::PauliX));

    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);

    // map |0> to |1>
    assert_eq!(0.0, result.output_state[0].re);
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(1.0, result.output_state[1].re);
    assert_eq!(0.0, result.output_state[1].im);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    let result = run(&input_state, Some(Gate::PauliX));

    assert_eq!(result.measurement, 0);
    assert_eq!(result.likelihood, 1.0);

    // map |1> to |0>
    assert_eq!(1.0, result.output_state[0].re);
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(0.0, result.output_state[1].re);
    assert_eq!(0.0, result.output_state[1].im);

    Ok(())
}

/// Prepare basis states |0> and |1> and:
/// - apply pauli-y gate [Wikipedia: Pauli Gates]
/// - measure
///
/// [Wikipedia: Pauli Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z))
#[test]
fn pauli_y() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    let result = run(&input_state, Some(Gate::PauliY));

    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);

    // map |0> to i|1>
    assert_eq!(0.0, result.output_state[0].re);
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(0.0, result.output_state[1].re);
    assert_eq!(1.0, result.output_state[1].im);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    let result = run(&input_state, Some(Gate::PauliY));

    assert_eq!(result.measurement, 0);
    assert_eq!(result.likelihood, 1.0);

    // map |1> to -i|0>
    assert_eq!(0.0, result.output_state[0].re);
    assert_eq!(-1.0, result.output_state[0].im);
    assert_eq!(0.0, result.output_state[1].re);
    assert_eq!(0.0, result.output_state[1].im);

    Ok(())
}

/// Prepare basis states |0> and |1> and:
/// - apply pauli-z gate [Wikipedia: Pauli Gates]
/// - measure
///
/// [Wikipedia: Pauli Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Pauli_gates_(X,Y,Z))
#[test]
fn pauli_z() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    let result = run(&input_state, Some(Gate::PauliZ));

    assert_eq!(result.measurement, 0);
    assert_eq!(result.likelihood, 1.0);

    // unchanged
    assert_eq!(input_state, result.output_state);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    let result = run(&input_state, Some(Gate::PauliZ));

    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);

    // map |1> to |-1>
    assert_eq!(0.0, result.output_state[0].re);
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(-1.0, result.output_state[1].re);
    assert_eq!(0.0, result.output_state[1].im);

    Ok(())
}

/// Prepare basis states |0> and |1> and:
/// - apply phase shifts(T, S) gates [Wikipedia: Phase Shift Gates]
/// - measure
///
/// [Wikipedia: Phase Shift Gates](https://en.wikipedia.org/wiki/Quantum_logic_gate#Phase_shift_gates)
#[test]
fn phase_shifts() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    // apply T gate
    let result = run(&input_state, Some(Gate::Phase(std::f64::consts::FRAC_PI_4)));

    // probabilities are unchanged
    assert_eq!(result.measurement, 0);
    assert_eq!(result.likelihood, 1.0);

    // map |0> to |0>
    // TODO: This is not working, research why.
    //assert_eq!(input_state, result.output_state);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    // apply T gate
    let result = run(&input_state, Some(Gate::Phase(std::f64::consts::FRAC_PI_4)));

    // probabilities are unchanged
    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);

    // map |1> to e^(i*theta)|-1>
    assert_ne!(input_state, result.output_state);

    // apply the inverse
    let result = run(
        &result.output_state,
        Some(Gate::Phase(-std::f64::consts::FRAC_PI_4)),
    );

    // probabilities are unchanged
    assert_eq!(result.measurement, 1);
    assert_eq!(result.likelihood, 1.0);

    // map to e^(i*theta)|-1> to |1>
    assert_eq!(input_state, result.output_state);

    Ok(())
}

/// Prepare basis states |0> and |1> and:
/// - apply hadamard gate [Wikipedia: Hadamard gate]
/// - measure
///
/// [Wikipedia: Hadamard gate](https://en.wikipedia.org/wiki/Quantum_logic_gate#Hadamard_gate)
#[test]
fn hadamard() -> Result<(), CircuitError> {
    // create initial state |0>
    let input_state = from_reals(&[1.0, 0.0]);

    // apply Hadamard gate
    let result = run(&input_state, Some(Gate::Hadamard));

    // map |0> to (|0> + |1>)/sqrt(2)
    // can't reliable get measurement as we are in superposition but can test
    // probabilities
    assert_almost_eq(result.likelihood, 0.5, 10);

    // we can also make sure the imaginary parts of the state are unchanged
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(0.0, result.output_state[1].im);

    // create initial state |1>
    let input_state = from_reals(&[0.0, 1.0]);

    // apply Hadamard gate
    let result = run(&input_state, Some(Gate::Hadamard));

    // map |1> to (|0> - |1>)/sqrt(2)
    // can't reliable get measurement as we are in superposition but can test
    // probabilities
    assert_almost_eq(result.likelihood, 0.5, 10);

    // we can also make sure the imaginary parts of the state are unchanged
    assert_eq!(0.0, result.output_state[0].im);
    assert_eq!(0.0, result.output_state[1].im);

    Ok(())
}
