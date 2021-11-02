// mod utils;
//
// use qip::pipeline::MeasurementHandle;
// use qip::qubits::RegisterHandle;
// use qip::*;
// use utils::assert_almost_eq;
//
// fn setup_cswap_sidechannel_circuit(
//     vec_n: u64,
// ) -> Result<(Register, RegisterHandle, RegisterHandle, MeasurementHandle), CircuitError> {
//     // Setup inputs
//     let mut b = OpBuilder::new();
//     let q1 = b.qubit();
//     let ra = b.register(vec_n)?;
//     let rb = b.register(vec_n)?;
//
//     // We will want to feed in some inputs later.
//     let ha = ra.handle();
//     let hb = rb.handle();
//
//     // Define circuit
//     let q1 = b.hadamard(q1);
//
//     // Make a qubit whose sole use is for sidechannels
//     let q2 = b.qubit();
//     let q2 = b.hadamard(q2);
//     let (_, h2) = b.measure(q2);
//
//     let mut c = b.with_condition(q1);
//     let _ = c.classical_sidechannel(
//         vec![ra, rb],
//         &[h2],
//         Box::new(|b, mut qs, _ms| {
//             let rb = qs.pop().unwrap();
//             let ra = qs.pop().unwrap();
//             let (ra, rb) = b.swap(ra, rb)?;
//             Ok(vec![ra, rb])
//         }),
//     );
//     let q1 = c.release_register();
//
//     let q1 = b.hadamard(q1);
//
//     let (q1, m1) = b.measure(q1);
//
//     Ok((q1, ha, hb, m1))
// }
//
// #[test]
// fn test_cswap_sidechannel() -> Result<(), CircuitError> {
//     let vec_n = 3;
//
//     let (q, ha, hb, m1) = setup_cswap_sidechannel_circuit(vec_n)?;
//
//     // Run circuit
//     let (_, measured) = run_local_with_init::<f64>(
//         &q,
//         &[ha.make_init_from_index(0)?, hb.make_init_from_index(0)?],
//     )?;
//
//     let (m, p) = measured.get_measurement(&m1).unwrap();
//     assert_eq!(m, 0);
//     assert_almost_eq(p, 1.0, 10);
//     Ok(())
// }
//
// #[test]
// fn test_cswap_sidechannel_unaligned() -> Result<(), CircuitError> {
//     let vec_n = 3;
//
//     let (q, ha, hb, m1) = setup_cswap_sidechannel_circuit(vec_n)?;
//
//     // Run circuit
//     let (_, measured) = run_local_with_init::<f64>(
//         &q,
//         &[ha.make_init_from_index(0)?, hb.make_init_from_index(1)?],
//     )?;
//
//     let (m, p) = measured.get_measurement(&m1).unwrap();
//     assert!(m == 0 || m == 1);
//     assert_almost_eq(p, 0.5, 10);
//     Ok(())
// }
