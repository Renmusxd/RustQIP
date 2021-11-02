use super::decomposition::decompose_unitary;
use crate::errors::CircuitError;
use crate::unitary_decomposition::decomposition::{BaseUnitary, DecompOp};
use crate::{condition, Complex, ConditionalContextBuilder, Register, UnitaryBuilder};
use num::{One, Zero};

/// Takes a unitary builder and a sparse unitary matrix and attempts to convert the matrix into the
/// equivalent circuit using basic gates. This is a bit numerically unstable and can be very
/// expensive for arbitrary matrices.
pub fn convert_sparse_to_circuit<U: UnitaryBuilder>(
    b: &mut U,
    r: Register,
    sparse_unitary: Vec<Vec<(u64, Complex<f64>)>>,
    drop_below: f64,
) -> Result<Register, CircuitError> {
    let decomposition = decompose_unitary(r.n(), sparse_unitary, drop_below)?;
    let (ops, base) =
        decomposition.map_err(|_| CircuitError::new("Decomposition failed.".to_string()))?;
    convert_decomp_ops_to_circuit(b, r, &base, &ops)
}

fn convert_decomp_ops_to_circuit<U: UnitaryBuilder>(
    b: &mut U,
    r: Register,
    base: &BaseUnitary<f64>,
    ops: &[DecompOp<f64>],
) -> Result<Register, CircuitError> {
    let n = r.n();
    let rs = b.split_all(r);

    let standard_mask = (1 << n) - 1;

    // Clear the correct index if it happens to be set
    let base_mask = base.top_row;
    let rs = negate_difference(b, rs, standard_mask, base_mask);

    let qubit_index = rs.len() as u64 - base.bit_index - 1;
    let rs = apply_to_index_with_control(b, rs, qubit_index, |cb, r| {
        cb.mat("Base", r, base.dat.to_vec()).unwrap()
    });
    let mask = base_mask;

    let (rs, mask) = ops.iter().fold((rs, mask), |(rs, mask), op| {
        match op {
            DecompOp::Phase { row, phi } => {
                let (row, phi) = (*row, *phi);
                let rs = negate_difference(b, rs, mask, row);

                // We can apply to any qubit.
                let rs = apply_to_index_with_control(b, rs, 0, |cb, q| {
                    let phase = Complex { re: 0.0, im: phi }.exp();
                    let phase_mat = vec![Complex::one(), Complex::zero(), Complex::zero(), phase];

                    let name = format!("Phase({:?})", phi);
                    cb.mat(&name, q, phase_mat).unwrap()
                });

                (rs, row)
            }
            DecompOp::Rotation {
                to_bits,
                bit_index,
                theta,
                ..
            } => {
                let new_mask = *to_bits;
                let rs = negate_difference(b, rs, mask, new_mask);

                let qubit_index = rs.len() as u64 - bit_index - 1;
                let rs = apply_to_index_with_control(b, rs, qubit_index, |cb, q| {
                    let (sin, cos) = theta.sin_cos();
                    let name = format!("Rotate({:?})", theta);
                    cb.real_mat(&name, q, &[cos, -sin, sin, cos]).unwrap()
                });
                (rs, new_mask)
            }
            DecompOp::Negate {
                row_b, bit_index, ..
            } => {
                let new_mask = *row_b;
                let rs = negate_difference(b, rs, mask, new_mask);

                let qubit_index = rs.len() as u64 - bit_index - 1;
                let rs = apply_to_index_with_control(b, rs, qubit_index, |cb, q| cb.not(q));

                (rs, new_mask)
            }
        }
    });
    let rs = negate_difference(b, rs, mask, !0);
    b.merge(rs)
}

fn apply_to_index_with_control<F, U>(
    b: &mut U,
    mut rs: Vec<Register>,
    indx: u64,
    f: F,
) -> Vec<Register>
where
    U: UnitaryBuilder,
    F: Fn(&mut ConditionalContextBuilder<U>, Register) -> Register,
{
    let r = rs.remove(indx as usize);
    let cr = b.merge(rs).unwrap();
    let (cr, r) = condition(b, cr, r, |b, r| f(b, r));
    let mut rs = b.split_all(cr);
    rs.insert(indx as usize, r);
    rs
}

fn negate_difference<U: UnitaryBuilder>(
    b: &mut U,
    rs: Vec<Register>,
    old_mask: u64,
    new_mask: u64,
) -> Vec<Register> {
    let needs_negation = old_mask ^ new_mask;
    (0..rs.len() as u64)
        .map(|indx| ((needs_negation >> indx) & 1) == 1)
        .rev()
        .zip(rs.into_iter())
        .map(|(negate, r)| if negate { b.not(r) } else { r })
        .collect()
}

#[cfg(test)]
mod unitary_decomp_circuit_tests {
    use super::*;
    use crate::pipeline::make_circuit_matrix;
    use crate::unitary_decomposition::test_utils::{
        flat_sparse, print_sparse, reconstruct_unitary,
    };
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
        let r = b.register(n)?;
        let r = convert_sparse_to_circuit(&mut b, r, v, EPSILON)?;

        // Output circuit in case it fails.
        run_debug(&r)?;
        let reconstructed = make_circuit_matrix::<f64>(n, &r, false);
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
            let r = b.register(n)?;
            let r = convert_decomp_ops_to_circuit(&mut b, r, &base, ops)?;

            run_debug(&r).unwrap();

            let circuit_mat = make_circuit_matrix::<f64>(n, &r, false);
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
