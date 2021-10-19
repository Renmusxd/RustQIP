/// Example of the Deutsch-Jozsa algorithm.
///
/// https://en.wikipedia.org/wiki/Deutsch%E2%80%93Jozsa_algorithm
/// https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html
use qip::{run_local, CircuitError, OpBuilder, Register, UnitaryBuilder};
use rand::Rng;

fn main() -> Result<(), CircuitError> {
    // Run the Deutsch-Jozsa algorithm for a constant function
    algorithm(FunctionType::Constant);
    // Run the Deutsch-Jozsa algorithm for a balanced function
    algorithm(FunctionType::Balanced);

    Ok(())
}

#[derive(Clone, Debug)]
enum FunctionType {
    Balanced,
    Constant,
}

/// Run a Deutsch-Jozsa algorithm given oracle function type.
fn algorithm(function_type: FunctionType) {
    let mut b = OpBuilder::new();

    println!("Input: Trying with a {:?} oracle", function_type);

    // in this example we always use a size of 3 for the bitstring.
    let n = 3;

    // first register with "n" qubits all in |0>
    let first_register = b.register(n as u64).unwrap();

    // second register with a single qubit and in state |1>
    let second_register = b.qubit();
    // the pauli-x gate will map |0> to |1>
    let second_register = b.x(second_register);

    // apply hadamard to all the qubits in the first register
    let first_register = b.hadamard(first_register);

    // apply oracle
    let (first_register, _second_register) = oracle(
        &mut b,
        first_register,
        second_register,
        function_type.clone(),
    );

    // apply hardamard to the first register
    let first_register = b.hadamard(first_register);

    // meassure the first register
    let (r, m) = b.measure(first_register);
    let (_, measurements) = run_local::<f64>(&r).unwrap();
    let measured = measurements.get_measurement(&m).unwrap();

    if measured.1 > 0.99 && measured.0 == 0 {
        // in a constant function the final state will be always 0 with
        // 100% probability.
        println!("Output: Function is {:?}", FunctionType::Constant);
    } else {
        // in any other scenario the function is balanced
        println!("Output: Function is {:?}", FunctionType::Balanced);
        // as we are here make sure that our oracle is really balanced
        test_balanced_oracle();
    }
}

/// Given first and second registers apply either a balanced or a constant
/// Deutsch-Jozsa oracle.
fn oracle(
    b: &mut OpBuilder,
    first: Register,
    second: Register,
    function_type: FunctionType,
) -> (Register, Register) {
    match function_type {
        FunctionType::Balanced => {
            let mut split: Vec<Register> = b.split_all(first).into_iter().collect();

            let mut vect = vec![];
            // Apply CNOT to each qubit in the first register using the second register qubit
            // as target. This is enough to create the most basic balanced function.
            let (first1, second) = b.cnot(split.pop().unwrap(), second);
            let (first2, second) = b.cnot(split.pop().unwrap(), second);
            let (first3, second) = b.cnot(split.pop().unwrap(), second);

            // Push inidividual qubits into a vector so we can merge them back into a register.
            vect.push(first1);
            vect.push(first2);
            vect.push(first3);
            let first = b.merge(vect).unwrap();

            (first, second)
        }
        FunctionType::Constant => {
            // Use a random number just to determine if this function will be
            // contant |0> or constant |1>
            let mut rng = rand::thread_rng();
            match rng.gen_range(0..2) {
                0 => (first, b.x(second)),
                _ => (first, second),
            }
        }
    }
}

/// Make sure our oracle is really balanced
fn test_balanced_oracle() {
    // all input states where the output will be 1
    let mut b = OpBuilder::new();
    let output = 1;

    // |000> -> 1
    let mut register = vec![];
    register.push(b.qubit());
    register.push(b.qubit());
    register.push(b.qubit());
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |011> -> 1
    let mut register = vec![];
    register.push(b.qubit());
    register.push(qubit_in_state_one(&mut b));
    register.push(qubit_in_state_one(&mut b));
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |101> -> 1
    let mut register = vec![];
    register.push(qubit_in_state_one(&mut b));
    register.push(b.qubit());
    register.push(qubit_in_state_one(&mut b));
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |110> -> 1
    let mut register = vec![];
    register.push(qubit_in_state_one(&mut b));
    register.push(qubit_in_state_one(&mut b));
    register.push(b.qubit());
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // all input states where the output will be 0
    let mut b = OpBuilder::new();
    let output = 0;

    // |001> -> 0
    let mut register = vec![];
    register.push(b.qubit());
    register.push(b.qubit());
    register.push(qubit_in_state_one(&mut b));
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |010> -> 0
    let mut register = vec![];
    register.push(b.qubit());
    register.push(qubit_in_state_one(&mut b));
    register.push(b.qubit());
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |100> -> 0
    let mut register = vec![];
    register.push(qubit_in_state_one(&mut b));
    register.push(b.qubit());
    register.push(b.qubit());
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);

    // |111> -> 0
    let mut register = vec![];
    register.push(qubit_in_state_one(&mut b));
    register.push(qubit_in_state_one(&mut b));
    register.push(qubit_in_state_one(&mut b));
    let first_register = b.merge(register).unwrap();
    test_oracle(&mut b, first_register, output);
}

// Given a first register and an output make sure the oracle outputs the right answer.
fn test_oracle(b: &mut OpBuilder, first_register: Register, output: u32) {
    // have a second register for output
    let second_register = b.qubit();
    let second_register = b.x(second_register);

    let (_first_register, second_register) =
        oracle(b, first_register, second_register, FunctionType::Balanced);

    // meassure the second register
    let (r, m) = b.measure(second_register);
    let (_, measurements) = run_local::<f64>(&r).unwrap();
    let measured = measurements.get_measurement(&m).unwrap();

    assert_eq!(measured.0, output as u64);
    assert_eq!(measured.1, 1.0);
}

/// Create a qubit in state |1>
fn qubit_in_state_one(b: &mut OpBuilder) -> Register {
    let qubit = b.qubit();
    b.x(qubit)
}
