extern crate qip;
extern crate rand;
use qip::pipeline::MeasurementHandle;
use qip::*;

fn run_alice(
    b: &mut OpBuilder,
    epr_alice: Qubit,
    initial_angle: f64,
) -> Result<MeasurementHandle, &'static str> {
    // Set up the qubits
    let q_random = b.qubit(1)?;

    // Create Alice's state
    let (sangle, cangle) = initial_angle.sin_cos();
    let q_random = b.real_mat("Rotate", q_random, &[cangle, -sangle, sangle, cangle])?;

    // Alice prepares her state: a|0> + b|1>
    let mut c = b.with_context(q_random);
    let q_alice = c.not(epr_alice);
    let q_random = c.release_qubit();
    let q_random = b.hadamard(q_random);

    // Now she measures her two particles
    let q = b.merge(vec![q_random, q_alice]);
    let (q, handle) = b.measure(q);

    Ok(handle)
}

fn run_bob(
    b: &mut OpBuilder,
    epr_bob: Qubit,
    handle: MeasurementHandle,
) -> Result<f64, &'static str> {
    let q_bob = b.single_qubit_classical_sidechannel(
        epr_bob,
        &[handle],
        Box::new(|b, q, measured| {
            // Based on the classical bits sent by Alice, Bob should apply a gate
            match measured {
                &[0b00] => Ok(q),
                &[0b10] => Ok(b.x(q)),
                &[0b01] => Ok(b.z(q)),
                &[0b11] => Ok(b.y(q)),
                _ => Err("Shouldn't be possible"),
            }
        }),
    );

    // Now Bob's qubit should be in Alice's state, let's check by faking some stochastic measurement
    let (q_bob, handle) = b.stochastic_measure(q_bob);
    let (_, mut measured) = run_local::<f64>(&q_bob)?;
    let ps = measured.pop_stochastic_measurements(handle).unwrap();

    // ps[0] = cos(theta)^2
    // ps[1] = sin(theta)^2
    // theta = atan(sqrt(ps[1]/ps[0]))
    Ok(ps[1].sqrt().atan2(ps[0].sqrt()))
}

fn main() -> Result<(), &'static str> {
    let random_angle = rand::random::<f64>() * std::f64::consts::FRAC_2_PI;

    let mut b = OpBuilder::new();
    let q_alice = b.qubit(1)?;
    let q_bob = b.qubit(1)?;

    // EPR pair
    let q_alice = b.hadamard(q_alice);
    let mut c = b.with_context(q_alice);
    let epr_bob = c.not(q_bob);
    let epr_alice = c.release_qubit();

    // Give Alice her EPR qubit
    let handle = run_alice(&mut b, epr_alice, random_angle)?;

    // Give Bob his and the classical measurements Alice made
    let teleported_angle = run_bob(&mut b, epr_bob, handle)?;

    println!(
        "Alice's angle: {:?}\tBob's angle: {:?}",
        random_angle, teleported_angle
    );

    Ok(())
}
