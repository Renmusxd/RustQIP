extern crate qip;
extern crate rand;
use qip::pipeline::LocalQuantumState;
use qip::*;

fn run_alice(initial_angle: f64) -> Result<(LocalQuantumState<f64>, u64), &'static str> {
    let mut b = OpBuilder::new();

    // Set up the qubits
    let q_random = b.qubit(1)?;
    let q_alice = b.qubit(1)?;
    let q_bob = b.qubit(1)?;

    // Create Alice's state
    let (sangle, cangle) = initial_angle.sin_cos();
    let q_random = b.real_mat("Rotate", q_random, &[cangle, -sangle, sangle, cangle])?;

    // EPR it
    let q_alice = b.hadamard(q_alice);
    let mut c = b.with_context(q_alice);
    let epr_bob = c.not(q_bob);
    let epr_alice = c.release_qubit();

    // Alice prepares her state: a|0> + b|1>
    let mut c = b.with_context(q_random);
    let q_alice = c.not(epr_alice);
    let q_random = c.release_qubit();
    let q_random = b.hadamard(q_random);

    // Now she measures her two particles
    let q = b.merge(vec![q_random, q_alice]);
    let (q, handle) = b.measure(q);

    let (state, measured) = run_local::<f64>(&q)?;
    let (m, _) = measured.get_measurement(&handle).unwrap();

    println!("Measured: {:02b}", m);

    Ok((state, m))
}

fn run_bob(state: LocalQuantumState<f64>, measured: u64) -> Result<f64, &'static str> {
    let mut b = OpBuilder::new();
    let q = b.qubit(3)?;
    let (q_bob, _) = b.split(q, vec![2])?;

    // Based on the classical bits sent by Alice, Bob should apply a gate
    let q_bob = match measured {
        0b00 => q_bob,
        0b10 => b.x(q_bob),
        0b01 => b.z(q_bob),
        0b11 => b.y(q_bob),
        _ => panic!("Shouldn't be possible"),
    };

    // Now Bob's qubit should be in Alice's state, let's check by faking some stochastic measurement
    let (q_bob, handle) = b.stochastic_measure(q_bob);
    let (_, mut measured) = run_with_state(&q_bob, state)?;
    let ps = measured.pop_stochastic_measurements(handle).unwrap();

    // ps[0] = cos(theta)^2
    // ps[1] = sin(theta)^2
    // theta = atan(sqrt(ps[1]/ps[0]))
    Ok(ps[1].sqrt().atan2(ps[0].sqrt()))
}

fn main() -> Result<(), &'static str> {
    // Can only measure angles between 0 and 90 degrees
    let random_angle = rand::random::<f64>() * std::f64::consts::FRAC_PI_2;

    let (state, measured) = run_alice(random_angle)?;
    let teleported_angle = run_bob(state, measured)?;

    println!(
        "Alice's angle: {:?}\tBob's angle: {:?}",
        random_angle, teleported_angle
    );

    Ok(())
}
