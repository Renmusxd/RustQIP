// mod utils;
//
// use qip::common_circuits::epr_pair;
// use qip::errors::CircuitError;
// use qip::pipeline::MeasurementHandle;
// use qip::*;
// use utils::assert_almost_eq;
//
// fn run_alice(b: &mut OpBuilder, epr_alice: Register, initial_angle: f64) -> MeasurementHandle {
//     // Set up the qubits
//     let q_random = b.qubit();
//
//     // Create Alice's state
//     let q_random = b.ry(q_random, initial_angle * 2.0);
//
//     // Alice prepares her state: a|0> + b|1>
//     let (q_random, q_alice) = b.cnot(q_random, epr_alice);
//     let q_random = b.hadamard(q_random);
//
//     // Now she measures her two particles
//     let q = b.merge(vec![q_random, q_alice]).unwrap();
//     let (_, handle) = b.measure(q);
//
//     handle
// }
//
// fn run_bob(
//     b: &mut OpBuilder,
//     epr_bob: Register,
//     handle: MeasurementHandle,
// ) -> Result<f64, CircuitError> {
//     let q_bob = b.single_register_classical_sidechannel(
//         epr_bob,
//         &[handle],
//         Box::new(|b, q, measured| {
//             // Based on the classical bits sent by Alice, Bob should apply a gate
//             match *measured {
//                 [0b00] => Ok(q),
//                 [0b10] => Ok(b.x(q)),
//                 [0b01] => Ok(b.z(q)),
//                 [0b11] => Ok(b.y(q)),
//                 _ => panic!("Shouldn't be possible"),
//             }
//         }),
//     );
//
//     // Now Bob's qubit should be in Alice's state, let's check by faking some stochastic measurement
//     let (q_bob, handle) = b.stochastic_measure(q_bob);
//
//     run_debug(&q_bob)?;
//
//     let (_, mut measured) = run_local::<f64>(&q_bob)?;
//     let ps = measured.pop_stochastic_measurements(handle).unwrap();
//
//     // ps[0] = cos(theta)^2
//     // ps[1] = sin(theta)^2
//     // theta = atan(sqrt(ps[1]/ps[0]))
//     Ok(ps[1].sqrt().atan2(ps[0].sqrt()))
// }
//
// #[test]
// fn test_teleport() -> Result<(), CircuitError> {
//     for i in 0..90 {
//         // Can only measure angles between 0 and 90 degrees
//         let random_angle = std::f64::consts::PI * i as f64 / 180.0;
//
//         let mut b = OpBuilder::new();
//         let (epr_alice, epr_bob) = epr_pair(&mut b, 1);
//
//         // Give Alice her EPR qubit
//         let handle = run_alice(&mut b, epr_alice, random_angle);
//
//         // Give Bob his and the classical measurements Alice made
//         let teleported_angle = run_bob(&mut b, epr_bob, handle)?;
//
//         assert_almost_eq(random_angle, teleported_angle, 10);
//     }
//     Ok(())
// }
