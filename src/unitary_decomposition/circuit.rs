use super::decomposition::decompose_unitary;
use crate::errors::CircuitError;
use crate::unitary_decomposition::decomposition::{BaseUnitary, DecompOp};
use crate::{Complex, Qubit, UnitaryBuilder};
use num::{One, Zero};

/// Takes a unitary builder and a sparse unitary matrix and attempts to convert the matrix into the
/// equivalent circuit using basic gates. This is a bit numerically unstable and can be very
/// expensive for arbitrary matrices.
pub fn convert_sparse_to_circuit(
    b: &mut dyn UnitaryBuilder,
    q: Qubit,
    sparse_unitary: Vec<Vec<(u64, Complex<f64>)>>,
    drop_below: f64,
) -> Result<Qubit, CircuitError> {
    let decomposition = decompose_unitary(q.n(), sparse_unitary, drop_below)?;
    let (ops, base) =
        decomposition.map_err(|_| CircuitError::new("Decomposition failed.".to_string()))?;
    convert_decomp_ops_to_circuit(b, q, &base, &ops)
}

fn convert_decomp_ops_to_circuit(
    b: &mut dyn UnitaryBuilder,
    q: Qubit,
    base: &BaseUnitary<f64>,
    ops: &[DecompOp<f64>],
) -> Result<Qubit, CircuitError> {
    let n = q.n();
    let qs = b.split_all(q);

    let standard_mask = (1 << n) - 1;

    // Clear the correct index if it happens to be set
    let base_mask = base.top_row;
    let qs = negate_difference(b, qs, standard_mask, base_mask);

    let qubit_index = qs.len() as u64 - base.bit_index - 1;
    let qs = apply_to_index_with_control(b, qs, qubit_index, |cb, q| {
        cb.mat("Base", q, base.dat.to_vec()).unwrap()
    });
    let mask = base_mask;

    let (qs, mask) = ops.iter().fold((qs, mask), |(qs, mask), op| {
        match op {
            DecompOp::Phase { row, phi } => {
                let (row, phi) = (*row, *phi);
                let qs = negate_difference(b, qs, mask, row);

                // We can apply to any qubit.
                let qs = apply_to_index_with_control(b, qs, 0, |cb, q| {
                    let phase = Complex { re: 0.0, im: phi }.exp();
                    let phase_mat = vec![Complex::one(), Complex::zero(), Complex::zero(), phase];

                    let name = format!("Phase({:?})", phi);
                    cb.mat(&name, q, phase_mat).unwrap()
                });

                (qs, row)
            }
            DecompOp::Rotation {
                from_bits,
                to_bits,
                bit_index,
                theta,
            } => {
                let new_mask = *to_bits;
                let qs = negate_difference(b, qs, mask, new_mask);

                let qubit_index = qs.len() as u64 - bit_index - 1;
                let qs = apply_to_index_with_control(b, qs, qubit_index, |cb, q| {
                    let (sin, cos) = theta.sin_cos();
                    let name = format!("Rotate({:?})", theta);
                    cb.real_mat(&name, q, &[cos, -sin, sin, cos]).unwrap()
                });
                (qs, new_mask)
            }
        }
    });
    let qs = negate_difference(b, qs, mask, !0);
    let q = b.merge(qs);
    Ok(q)
}

fn apply_to_index_with_control<F: Fn(&mut dyn UnitaryBuilder, Qubit) -> Qubit>(
    b: &mut dyn UnitaryBuilder,
    mut qs: Vec<Qubit>,
    indx: u64,
    f: F,
) -> Vec<Qubit> {
    let q = qs.remove(indx as usize);
    let cq = b.merge(qs);
    let mut cb = b.with_condition(cq);
    let q = f(&mut cb, q);
    let cq = cb.release_qubit();
    let mut qs = b.split_all(cq);
    qs.insert(indx as usize, q);
    qs
}

fn negate_difference(
    b: &mut dyn UnitaryBuilder,
    qs: Vec<Qubit>,
    old_mask: u64,
    new_mask: u64,
) -> Vec<Qubit> {
    let needs_negation = old_mask ^ new_mask;
    (0..qs.len() as u64)
        .map(|indx| ((needs_negation >> indx) & 1) == 1)
        .rev()
        .zip(qs.into_iter())
        .map(|(negate, q)| if negate { b.not(q) } else { q })
        .collect()
}

#[cfg(test)]
mod unitary_decomp_circuit_tests {
    use super::*;
    use crate::pipeline::make_circuit_matrix;
    use crate::unitary_decomposition::decomposition::reconstruct_unitary;
    use crate::unitary_decomposition::utils::{flat_sparse, print_sparse};
    use crate::{run_debug, OpBuilder, Precision};
    use std::error::Error;

    const EPSILON: f64 = 0.00000000001;

    fn sparse_from_reals<P: Precision>(v: Vec<Vec<(u64, P)>>) -> Vec<Vec<(u64, Complex<P>)>> {
        let v = v
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|(col, r)| (col, (r, P::zero())))
                    .collect()
            })
            .collect();
        sparse_from_tuples(v)
    }

    fn sparse_from_tuples<P: Precision>(v: Vec<Vec<(u64, (P, P))>>) -> Vec<Vec<(u64, Complex<P>)>> {
        v.into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|(col, (a, b))| (col, Complex { re: a, im: b }))
                    .collect()
            })
            .collect()
    }

    fn flat_round(v: Vec<Vec<(u64, Complex<f64>)>>, prec: i32) -> Vec<(u64, u64, Complex<f64>)> {
        let flat = flat_sparse(v);
        flat.into_iter()
            .map(|(row, col, val)| {
                let p = 10.0f64.powi(prec);
                let val = Complex {
                    re: (val.re * p).round() / p,
                    im: (val.im * p).round() / p,
                };
                (row, col, val)
            })
            .filter(|(_, _, v)| *v != Complex::zero())
            .collect()
    }

    fn assert_decomp(n: u64, v: Vec<Vec<(u64, Complex<f64>)>>) -> Result<(), CircuitError> {
        let flat_v = flat_round(v.clone(), 10);
        let mut b = OpBuilder::new();
        let q = b.qubit(n).unwrap();
        let q = convert_sparse_to_circuit(&mut b, q, v, EPSILON)?;

        // Output circuit in case it fails.
        run_debug(&q)?;
        let reconstructed = make_circuit_matrix::<f64>(n, &q, false);
        let reconstructed = reconstructed
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .enumerate()
                    .map(|(indx, c)| (indx as u64, c))
                    .collect()
            })
            .collect();
        let flat_reconstructed = flat_round(reconstructed, 10);
        assert_eq!(flat_reconstructed, flat_v);
        Ok(())
    }

    fn assert_decomp_ops_and_base(
        n: u64,
        base: Option<BaseUnitary<f64>>,
        ops: Vec<DecompOp<f64>>,
    ) -> Result<(), CircuitError> {
        let base = base.unwrap_or(BaseUnitary {
            top_row: (1 << n) - 1,
            bot_row: (1 << n) - 2,
            bit_index: 0,
            dat: [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::one(),
            ],
        });

        println!("Base: {:?}", base);
        println!("Ops: {:?}", ops);

        (0..ops.len()).try_for_each(|indx| {
            let ops = &ops[..indx];

            let reconstructed = reconstruct_unitary(n, ops, &base);

            let mut b = OpBuilder::new();
            let q = b.qubit(n)?;
            let q = convert_decomp_ops_to_circuit(&mut b, q, &base, ops)?;

            run_debug(&q).unwrap();

            let circuit_mat = make_circuit_matrix::<f64>(n, &q, false);
            let circuit_mat: Vec<_> = circuit_mat
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .enumerate()
                        .map(|(col, c)| (col as u64, c))
                        .filter(|(_, c)| *c != Complex::zero())
                        .collect()
                })
                .collect();

            println!("Expected:");
            print_sparse(&reconstructed);
            println!("Found:");
            print_sparse(&circuit_mat);

            let flat_reconstructed = flat_round(reconstructed, 10);
            let flat_circuit_mat = flat_round(circuit_mat, 10);
            assert_eq!(flat_reconstructed, flat_circuit_mat);
            Ok(())
        })
    }

    #[test]
    fn test_identity_rotate_decomp_ops() -> Result<(), Box<dyn Error>> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests.into_iter().try_for_each(
            |(from_bits, to_bits, bit_index)| -> Result<(), Box<dyn Error>> {
                let ops = vec![DecompOp::Rotation {
                    from_bits,
                    to_bits,
                    bit_index,
                    theta: 0.0,
                }];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            },
        )?;
        Ok(())
    }

    #[test]
    fn test_identity_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![0, 1, 2, 3];
        tests.into_iter().try_for_each(|row| {
            let ops = vec![DecompOp::Phase { row, phi: 0.0 }];
            assert_decomp_ops_and_base(2, None, ops)?;
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_half_rotate_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![DecompOp::Rotation {
                    from_bits,
                    to_bits,
                    bit_index,
                    theta: std::f64::consts::PI,
                }];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_half_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![0, 1, 2, 3];
        tests.into_iter().try_for_each(|row| {
            let ops = vec![DecompOp::Phase {
                row,
                phi: std::f64::consts::PI,
            }];
            assert_decomp_ops_and_base(2, None, ops)?;
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_quarter_rotate_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![DecompOp::Rotation {
                    from_bits,
                    to_bits,
                    bit_index,
                    theta: std::f64::consts::FRAC_PI_2,
                }];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_quarter_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![0, 1, 2, 3];
        tests.into_iter().try_for_each(|row| {
            let ops = vec![DecompOp::Phase {
                row,
                phi: std::f64::consts::FRAC_PI_2,
            }];
            assert_decomp_ops_and_base(2, None, ops)?;
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_two_quarter_rotate_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                ];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_two_quarter_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![0, 1, 2, 3];
        tests.into_iter().try_for_each(|row| {
            let ops = vec![
                DecompOp::Phase {
                    row,
                    phi: std::f64::consts::FRAC_PI_2,
                },
                DecompOp::Phase {
                    row,
                    phi: std::f64::consts::FRAC_PI_2,
                },
            ];
            assert_decomp_ops_and_base(2, None, ops)?;
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_undo_quarter_rotate_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: -std::f64::consts::FRAC_PI_2,
                    },
                ];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_undo_quarter_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![0, 1, 2, 3];
        tests.into_iter().try_for_each(|row| {
            let ops = vec![
                DecompOp::Phase {
                    row,
                    phi: std::f64::consts::FRAC_PI_2,
                },
                DecompOp::Phase {
                    row,
                    phi: -std::f64::consts::FRAC_PI_2,
                },
            ];
            assert_decomp_ops_and_base(2, None, ops)?;
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn test_undo_quarter_row_rotate_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                    DecompOp::Rotation {
                        to_bits,
                        from_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                ];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_sequence_rotate_phase_decomp_ops() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                    DecompOp::Phase {
                        row: to_bits,
                        phi: std::f64::consts::FRAC_PI_2,
                    },
                ];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_sequence_rotate_phase_decomp_ops_2() -> Result<(), CircuitError> {
        let tests = vec![
            (0b00, 0b10, 1),
            (0b00, 0b01, 0),
            (0b01, 0b11, 1),
            (0b01, 0b00, 0),
            (0b10, 0b00, 1),
            (0b10, 0b11, 0),
            (0b11, 0b01, 1),
            (0b11, 0b10, 0),
        ];
        tests
            .into_iter()
            .try_for_each(|(from_bits, to_bits, bit_index)| {
                let ops = vec![
                    DecompOp::Rotation {
                        from_bits,
                        to_bits,
                        bit_index,
                        theta: std::f64::consts::FRAC_PI_2,
                    },
                    DecompOp::Phase {
                        row: from_bits,
                        phi: std::f64::consts::FRAC_PI_2,
                    },
                ];
                assert_decomp_ops_and_base(2, None, ops)?;
                Ok(())
            })?;
        Ok(())
    }

    #[test]
    fn test_decompose_basic() -> Result<(), CircuitError> {
        let v = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        assert_decomp(2, v)?;
        Ok(())
    }

    #[test]
    fn test_decompose_rotation() -> Result<(), CircuitError> {
        let (s, c) = std::f64::consts::FRAC_PI_8.sin_cos();
        let v = vec![
            vec![(0, c), (2, -s)],
            vec![(1, 1.0)],
            vec![(0, s), (2, c)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        assert_decomp(2, v)?;
        Ok(())
    }

    #[test]
    fn test_decompose_pauli() -> Result<(), CircuitError> {
        let v = vec![
            vec![
                (0, (std::f64::consts::FRAC_1_SQRT_2, 0.0)),
                (1, (0.0, -std::f64::consts::FRAC_1_SQRT_2)),
            ],
            vec![
                (0, (0.0, -std::f64::consts::FRAC_1_SQRT_2)),
                (1, (std::f64::consts::FRAC_1_SQRT_2, 0.0)),
            ],
            vec![(2, (1.0, 0.0))],
            vec![(3, (1.0, 0.0))],
        ];
        let v = sparse_from_tuples(v);
        assert_decomp(2, v)?;
        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_decomp() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, Complex::one())])
            .collect();
        assert_decomp(n, v)?;
        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_imag_decomp() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, (0.0, 1.0))])
            .collect();
        let v = sparse_from_tuples(v);
        assert_decomp(n, v)?;
        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_decomp_iterative() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, Complex::one())])
            .collect();

        let (ops, base) = decompose_unitary(n, v, 1e-10)?
            .map_err(|_| CircuitError::new("Decomposition Failed".to_string()))?;

        assert_decomp_ops_and_base(n, Some(base), ops)?;

        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_imag_decomp_iterative() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, (0.0, 1.0))])
            .collect();
        let v = sparse_from_tuples(v);

        let (ops, base) = decompose_unitary(n, v, 1e-10)?
            .map_err(|_| CircuitError::new("Decomposition Failed".to_string()))?;

        assert_decomp_ops_and_base(n, Some(base), ops)?;

        Ok(())
    }
}
