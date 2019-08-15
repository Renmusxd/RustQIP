use crate::errors::CircuitError;
use crate::unitary_decomposition::bit_pathing::BitPather;
use crate::unitary_decomposition::utils::*;
use crate::Complex;
use crate::Precision;
use num::{One, Zero};

/// A controlled phase or rotation op.
#[derive(Debug)]
pub enum DecompOp<P: Precision> {
    /// A phase rotation on all |row> entries.
    Phase {
        /// row to phase shift
        row: u64,
        /// Phase shift to apply `e^{i*phi}`
        phi: P,
    },
    /// An op which rotates with `c|from><from| + s|to><from| - s|from><to| + c|to><to| + I`
    Rotation {
        /// Bits to rotates away from
        from_bits: u64,
        /// Bit to rotate to
        to_bits: u64,
        /// Index of rotated bit
        bit_index: u64,
        /// Angle to rotate
        theta: P,
    },
}

/// A base controlled single qubit op.
#[derive(Debug)]
pub struct BaseUnitary<P: Precision> {
    /// row index for top entries `a` and `b`
    pub top_row: u64,
    /// row index for bot entries `c` and `d`.
    pub bot_row: u64,
    /// Index of bit difference between top and bot
    pub bit_index: u64,
    /// Data of base `[[a,b],[c,d]]`
    pub dat: [Complex<P>; 4],
}

impl<P: Precision> Default for BaseUnitary<P> {
    fn default() -> Self {
        BaseUnitary {
            top_row: 0,
            bot_row: 1,
            bit_index: 0,
            dat: [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::one(),
            ],
        }
    }
}

/// Use the ops and base unitary to reconstruct the decomposed unitary op.
pub fn reconstruct_unitary<P: Precision + Clone>(
    n: u64,
    ops: &[DecompOp<P>],
    base: &BaseUnitary<P>,
) -> Vec<Vec<(u64, Complex<P>)>> {
    let keep_threshold = P::from(1e-10).unwrap();
    let mut base_mat: Vec<_> = (0..1 << n)
        .map(|indx| {
            if indx == base.top_row {
                vec![(base.top_row, base.dat[0]), (base.bot_row, base.dat[1])]
            } else if indx == base.bot_row {
                vec![(base.top_row, base.dat[2]), (base.bot_row, base.dat[3])]
            } else {
                vec![(indx, Complex::one())]
            }
        })
        .collect();
    base_mat[base.top_row as usize].sort_by_key(|(col, _)| *col);
    base_mat[base.top_row as usize].retain(|(_, val)| val.norm_sqr() >= keep_threshold);
    base_mat[base.bot_row as usize].sort_by_key(|(col, _)| *col);
    base_mat[base.bot_row as usize].retain(|(_, val)| val.norm_sqr() >= keep_threshold);

    ops.iter().for_each(|op| {
        match op {
            DecompOp::Rotation {
                from_bits,
                to_bits,
                theta,
                ..
            } => apply_controlled_rotation_and_clean(
                *from_bits,
                *to_bits,
                *theta,
                &mut base_mat,
                |val| val.norm_sqr() >= keep_threshold,
            ),
            DecompOp::Phase { row, phi } => {
                apply_phase_to_row(*phi, &mut base_mat[(*row) as usize])
            }
        };
    });
    base_mat
}

fn consolidate_column<P: Precision>(
    pathfinder: &BitPather,
    column: u64,
    sparse_mat: &mut [Vec<(u64, Complex<P>)>],
    keep_threshold: P,
) -> Result<Vec<DecompOp<P>>, CircuitError> {
    let nonzeros = sparse_mat
        .iter()
        .enumerate()
        .fold(vec![], |mut nonzeros, (indx, row)| {
            let val = sparse_value_at_col(column, row);
            match val {
                Some(_) => {
                    nonzeros.push(indx as u64);
                    nonzeros
                }
                None => nonzeros,
            }
        });

    let results = pathfinder.path(column, &nonzeros)?.into_iter().try_fold(
        vec![],
        |mut ops, (from_code, to_code)| {
            let from_row = &sparse_mat[from_code as usize];
            let to_row = &sparse_mat[to_code as usize];

            let from_val = sparse_value_at_col(column, from_row)
                .copied()
                .unwrap_or_default();
            let to_val = sparse_value_at_col(column, to_row)
                .copied()
                .unwrap_or_default();

            let (from_r, from_phi) = from_val.to_polar();
            let (to_r, _) = to_val.to_polar();

            // now we have a(to)e^{i*phi}|to> + a(from)|from>
            // now we have a*sin(theta)*e^{i*phi}|to> + a*cos(theta)|from>
            // a = sqrt(|a(to)|^2 + |a(from)|^2)
            // cos(theta) = |a(to)|/a
            // sin(theta) = |a(from)|/a
            // Therefore: theta = atan(|a(from)|/|a(to)|)
            let theta = from_r.atan2(to_r);
            if from_phi != P::zero() {
                apply_phase_to_row(-from_phi, &mut sparse_mat[from_code as usize]);
                ops.push(DecompOp::Phase {
                    row: from_code,
                    phi: from_phi,
                });
            }

            if theta != P::zero() {
                apply_controlled_rotation_and_clean(from_code, to_code, theta, sparse_mat, |val| {
                    val.norm_sqr() >= keep_threshold
                });

                let mut mask = from_code ^ to_code;
                let mut mask_index = 0;
                mask >>= 1;
                while mask > 0 {
                    mask >>= 1;
                    mask_index += 1;
                }
                ops.push(DecompOp::Rotation {
                    from_bits: to_code,
                    to_bits: from_code,
                    bit_index: mask_index,
                    theta,
                });
            }

            let phi = sparse_value_at_coords(to_code as usize, column, &sparse_mat)
                .map(|c| c.to_polar().1)
                .unwrap_or_else(P::zero);
            if phi != P::zero() {
                apply_phase_to_row(-phi, &mut sparse_mat[to_code as usize]);
                ops.push(DecompOp::Phase { row: to_code, phi });
            }

            Ok(ops)
        },
    )?;

    // Try to catch failures early, it's an expensive operation for each column.
    if sparse_mat[column as usize].len() != 1 {
        let row_mag = row_magnitude_sqr(column, sparse_mat);
        let col_mag = column_magnitude_sqr(column, sparse_mat);
        let entry_mag = sparse_value_at_coords(column as usize, column, sparse_mat)
            .map(|v| v.norm_sqr())
            .unwrap_or_else(P::zero);

        let message = format!(
            "Could not consolidate col/row = {:?}. |mat[col,col]|={}. |column|={}. |row|={}",
            column, entry_mag, col_mag, row_mag
        );
        CircuitError::make_err(message)
    } else {
        Ok(results)
    }
}

/// A successful decomposition with the list of ops to recreate the matrix, and a base controlled
/// single qubit op.
pub type DecompositionSuccess<P> = (Vec<DecompOp<P>>, BaseUnitary<P>);
/// An unsuccessful decomposition with the list of ops applied and the remaining sparse matrix which
/// is not a controlled single qubit op.
pub type DecompositionFailure<P> = (Vec<DecompOp<P>>, Vec<Vec<(u64, Complex<P>)>>);
/// The result of a decomposition.
pub type DecompositionResult<P> = Result<DecompositionSuccess<P>, DecompositionFailure<P>>;

/// Decompose the unitary op (represented as a vector of sparse column/values).
/// This uses the row-by-row rotation algorithm in gray-coding space to move weights from `|xj>` to
/// `|xi>`, this can, in worst case, use `2^2n` gates for `n` qubits.
/// Need a good algorithm for consolidating entries in sparse unitary matrices other than iterating
/// through all the gray codes, this is basically a Steiner tree on a graph where the graph is the
/// vertices of a n-dimensional hypercube, it just so happens a paper was written on this:
/// https://www.researchgate.net/publication/220617458_Near_Optimal_Bounds_for_Steiner_Trees_in_the_Hypercube
pub fn decompose_unitary<P: Precision>(
    n: u64,
    mut sparse_mat: Vec<Vec<(u64, Complex<P>)>>,
    drop_below_mag: P,
) -> Result<DecompositionResult<P>, CircuitError> {
    let keep_threshold = drop_below_mag.powi(2);
    // Get the order in which we should consolidate entries.
    let mut encoding = gray_code(n);
    let last = encoding.pop().unwrap();
    let second_last = encoding.pop().unwrap();

    let pathfinder = BitPather::new(n);
    let mut ops = encoding.into_iter().try_fold(vec![], |mut ops, target| {
        let new_ops = consolidate_column(&pathfinder, target, &mut sparse_mat, keep_threshold)?;
        ops.extend(new_ops);
        Ok(ops)
    })?;
    ops.reverse();

    let second_last_row = &sparse_mat[second_last as usize];
    let last_row = &sparse_mat[last as usize];
    let dat_a = second_last_row
        .binary_search_by_key(&second_last, |(c, _)| *c)
        .map(|indx| second_last_row[indx].1)
        .unwrap_or_default();
    let dat_b = second_last_row
        .binary_search_by_key(&last, |(c, _)| *c)
        .map(|indx| second_last_row[indx].1)
        .unwrap_or_default();
    let dat_c = last_row
        .binary_search_by_key(&second_last, |(c, _)| *c)
        .map(|indx| last_row[indx].1)
        .unwrap_or_default();
    let dat_d = last_row
        .binary_search_by_key(&last, |(c, _)| *c)
        .map(|indx| last_row[indx].1)
        .unwrap_or_default();

    // Check if decomposition was successful
    let result = sparse_mat
        .iter()
        .enumerate()
        .try_for_each(|(indx, row)| -> Result<(), ()> {
            let indx = indx as u64;
            if row.len() == 1 && row[0].0 == indx {
                Ok(())
            } else if indx != second_last && indx != last {
                Err(())
            } else {
                Ok(())
            }
        });

    let result = match result {
        Ok(()) => {
            let mut mask_index = 0;
            let mut mask = last ^ second_last;
            mask >>= 1;
            while mask > 0 {
                mask >>= 1;
                mask_index += 1;
            }

            Ok((
                ops,
                BaseUnitary {
                    top_row: second_last,
                    bot_row: last,
                    bit_index: mask_index,
                    dat: [dat_a, dat_b, dat_c, dat_d],
                },
            ))
        }
        Err(()) => Err((ops, sparse_mat)),
    };

    Ok(result)
}

#[cfg(test)]
mod unitary_decomp_tests {
    use super::*;
    use num::Zero;

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

    #[test]
    fn test_basic_decomp() -> Result<(), CircuitError> {
        let v = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(2, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose matrix".to_string()))?;
        let rebuilt = reconstruct_unitary(2, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_rotated_decomp() -> Result<(), CircuitError> {
        let v = vec![
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (1, -std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (1, std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(2, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(2, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_rotated_other_decomp() -> Result<(), CircuitError> {
        let v = vec![
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (2, -std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(1, 1.0)],
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (2, std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(2, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(2, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_rotated_eighth_decomp() -> Result<(), CircuitError> {
        let (s, c) = std::f64::consts::FRAC_PI_8.sin_cos();
        let v = vec![
            vec![(0, c), (2, -s)],
            vec![(1, 1.0)],
            vec![(0, s), (2, c)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(2, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(2, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_pauli_decomp() -> Result<(), CircuitError> {
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
        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(2, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(2, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_decomp() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, Complex::one())])
            .collect();

        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(n, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(n, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }

    #[test]
    fn test_larger_antidiagonal_imag_decomp() -> Result<(), CircuitError> {
        let n = 3;
        let v: Vec<_> = (0..1 << n)
            .map(|row| vec![((1 << n) - row - 1, (0.0, 1.0))])
            .collect();
        let v = sparse_from_tuples(v);

        let flat_v = flat_round(v.clone(), 10);

        let (ops, base) = decompose_unitary(n, v, EPSILON)?
            .map_err(|_| CircuitError::new("Failed to decompose".to_string()))?;
        let rebuilt = reconstruct_unitary(n, &ops, &base);

        let flat_r = flat_round(rebuilt.clone(), 10);
        assert_eq!(flat_v, flat_r);
        Ok(())
    }
}
