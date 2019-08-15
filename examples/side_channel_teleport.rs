extern crate qip;
extern crate rand;
use qip::common_circuits::epr_pair;
use qip::pipeline::MeasurementHandle;
use qip::*;

fn run_alice(
    b: &mut OpBuilder,
    epr_alice: Register,
    initial_angle: f64,
) -> Result<MeasurementHandle, CircuitError> {
    // Set up the qubits
    let q_random = b.register(1)?;

    // Create Alice's state
    let (sangle, cangle) = initial_angle.sin_cos();
    let q_random = b.real_mat("Rotate", q_random, &[cangle, -sangle, sangle, cangle])?;

    // Alice prepares her state: a|0> + b|1>
    let (q_random, q_alice) = condition(b, q_random, epr_alice, |c, q| Ok(c.not(q)))?;
    let q_random = b.hadamard(q_random);

    // Now she measures her two particles
    let q = b.merge(vec![q_random, q_alice]);
    let (q, handle) = b.measure(q);

    Ok(handle)
}

fn run_bob(
    b: &mut OpBuilder,
    epr_bob: Register,
    handle: MeasurementHandle,
) -> Result<f64, CircuitError> {
    let q_bob = b.single_register_classical_sidechannel(
        epr_bob,
        &[handle],
        Box::new(|b, q, measured| {
            // Based on the classical bits sent by Alice, Bob should apply a gate
            match measured {
                &[0b00] => Ok(q),
                &[0b10] => Ok(b.x(q)),
                &[0b01] => Ok(b.z(q)),
                &[0b11] => Ok(b.y(q)),
                _ => panic!("Shouldn't be possible"),
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

fn main() -> Result<(), CircuitError> {
    // Can only measure angles between 0 and 90 degrees
    let random_angle = rand::random::<f64>() * std::f64::consts::FRAC_PI_2;

    let mut b = OpBuilder::new();

    // EPR pair
    let (epr_alice, epr_bob) = epr_pair(&mut b, 1);

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
