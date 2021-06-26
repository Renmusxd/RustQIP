extern crate num;
extern crate qip;

mod utils;

use qip::{state_ops::from_reals, *};

// Classic and quantum representation of two-qubit initial states.
struct InitialState {
    first_bit: u8,
    first_qubit: Vec<Complex<f64>>,
    second_bit: u8,
    second_qubit: Vec<Complex<f64>>,
}

// Creates the four computational basis states for two-qubit.
fn initial_states() -> Vec<InitialState> {
    let basic_state_zero = from_reals(&[1.0, 0.0]);
    let basic_state_one = from_reals(&[0.0, 1.0]);

    vec![
        // |00>
        InitialState {
            first_bit: 0,
            first_qubit: basic_state_zero.clone(),
            second_bit: 0,
            second_qubit: basic_state_zero.clone(),
        },
        // |01>
        InitialState {
            first_bit: 0,
            first_qubit: basic_state_zero.clone(),
            second_bit: 1,
            second_qubit: basic_state_one.clone(),
        },
        // |10>
        InitialState {
            first_bit: 1,
            first_qubit: basic_state_one.clone(),
            second_bit: 0,
            second_qubit: basic_state_zero.clone(),
        },
        // |11>
        InitialState {
            first_bit: 1,
            first_qubit: basic_state_one.clone(),
            second_bit: 1,
            second_qubit: basic_state_one.clone(),
        },
    ]
}

#[test]
fn create_entanglement() -> Result<(), CircuitError> {
    let basis_inputs = initial_states();

    for input in basis_inputs {
        let mut b = OpBuilder::new();
        let q1 = b.qubit();
        let q2 = b.qubit();

        let h1 = q1.handle();
        let h2 = q2.handle();

        let initial_state = [
            h1.make_init_from_state(vec![input.first_qubit[0], input.first_qubit[1]])
                .unwrap(),
            h2.make_init_from_state(vec![input.second_qubit[0], input.second_qubit[1]])
                .unwrap(),
        ];

        // entangle q1 and q2
        let q1 = b.hadamard(q1);
        let (q1, q2) = b.cnot(q1, q2);

        // merge
        let q = b.merge(vec![q1, q2]).unwrap();

        // measure the merged qubit
        let (q, m) = b.measure(q);

        // run and get measurment
        let (_, measurements) = run_local_with_init::<f64>(&q, &initial_state).ok().unwrap();
        let (m, l) = measurements.get_measurement(&m).unwrap();

        // We can't test the measure result as the qubit is in an entangled state but
        // we know the likelihood of getting one of two possible states is 50-50.
        assert!(m == 0 || m <= 3);
        utils::assert_almost_eq(l, 0.5, 10);
        // TODO: Can we do something better here ?
    }

    Ok(())
}

#[test]
fn measure_entanglement() -> Result<(), CircuitError> {
    let basis_inputs = initial_states();

    for input in basis_inputs {
        let mut b = OpBuilder::new();
        let q1 = b.qubit();
        let q2 = b.qubit();

        let h1 = q1.handle();
        let h2 = q2.handle();

        let initial_state = [
            h1.make_init_from_state(vec![input.first_qubit[0], input.first_qubit[1]])
                .unwrap(),
            h2.make_init_from_state(vec![input.second_qubit[0], input.second_qubit[1]])
                .unwrap(),
        ];

        // entangle q1 and q2
        let q1 = b.hadamard(q1);
        let (q1, q2) = b.cnot(q1, q2);

        // apply the reverse and merge
        let (q1, q2) = b.cnot(q1, q2);
        let q1 = b.hadamard(q1);
        let q = b.merge(vec![q1, q2]).unwrap();

        // measure the merged qubit
        let (q, m) = b.measure(q);

        // run and get measurment
        let (_, measurements) = run_local_with_init::<f64>(&q, &initial_state).ok().unwrap();
        let (m, l) = measurements.get_measurement(&m).unwrap();

        // likelihood is always 1.0
        utils::assert_almost_eq(l, 1.0, 10);

        // depending on the measurment result we can know with no ambiguity the initial state.
        let binary_m = format!("{:02b}", m);
        assert_eq!(
            binary_m.chars().nth(0).unwrap().to_digit(2).unwrap(),
            input.second_bit as u32
        );
        assert_eq!(
            binary_m.chars().nth(1).unwrap().to_digit(2).unwrap(),
            input.first_bit as u32
        );
        // TODO: Positions are reversed, check if there is some sort of endianess happening.
    }

    Ok(())
}
