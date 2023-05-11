use crate::iterators::{num_indices, sum_for_ops_cols, MatrixOp};
use crate::utils::{get_bit, set_bit};
use crate::{iter, iter_mut};
use num_traits::{One, Zero};
use std::iter::Sum;
use std::ops::Mul;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Get the index for a submatrix indexed by `indices` given the `full_index` for the larger 2^n by 2^n matrix.
pub fn full_to_sub(n: usize, mat_indices: &[usize], full_index: usize) -> usize {
    let nindices = mat_indices.len();
    mat_indices
        .iter()
        .enumerate()
        .fold(0, |acc, (j, indx)| -> usize {
            let bit = get_bit(full_index, n - 1 - *indx);
            set_bit(acc, nindices - 1 - j, bit)
        })
}

/// Given the `sub_index` for the submatrix, and a base to overwrite values, get the full index for the 2^n by 2^n matrix.
pub fn sub_to_full(n: usize, mat_indices: &[usize], sub_index: usize, base: usize) -> usize {
    let nindices = mat_indices.len();
    mat_indices.iter().enumerate().fold(base, |acc, (j, indx)| {
        let bit = get_bit(sub_index, nindices - 1 - j);
        set_bit(acc, n - 1 - *indx, bit)
    })
}

/// Get the `i`th qubit index for `op`
pub fn get_index<P>(op: &MatrixOp<P>, i: usize) -> usize {
    match &op {
        MatrixOp::Matrix(indices, _) => indices[i],
        MatrixOp::SparseMatrix(indices, _) => indices[i],
        MatrixOp::Swap(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        MatrixOp::Control(cs, os, _) => {
            if i < cs.len() {
                cs[i]
            } else {
                os[i - cs.len()]
            }
        }
    }
}

/// Apply `op` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
pub fn apply_op<P>(
    n: usize,
    op: &MatrixOp<P>,
    input: &[P],
    output: &mut [P],
    input_offset: usize,
    output_offset: usize,
) where
    P: Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    let mat_indices: Vec<usize> = (0..num_indices(op)).map(|i| get_index(op, i)).collect();
    let nindices = mat_indices.len();

    let row_fn = |(outputrow, outputloc): (usize, &mut P)| {
        let row = output_offset + (outputrow);
        let matrow = full_to_sub(n, &mat_indices, row);
        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
        // for the row calculated above.
        let f = |(i, val): (usize, P)| -> P {
            let colbits = sub_to_full(n, &mat_indices, i, row);
            if colbits < input_offset {
                P::zero()
            } else {
                let vecrow = colbits - input_offset;
                if vecrow >= input.len() {
                    P::zero()
                } else {
                    val * input[vecrow].clone()
                }
            }
        };

        // Get value for row and assign
        *outputloc = op.sum_for_op_cols(nindices, matrow, f);
    };

    // Generate output for each output row
    iter_mut!(output).enumerate().for_each(row_fn);
}

/// Apply `ops` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
/// This is much less efficient as compared to repeated applications of `apply_op`, if your ops can
/// be applied in sequence, do so with `apply_op`.
pub fn apply_ops<P>(
    n: usize,
    ops: &[MatrixOp<P>],
    input: &[P],
    output: &mut [P],
    input_offset: usize,
    output_offset: usize,
) where
    P: Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    match ops {
        [op] => apply_op(n, op, input, output, input_offset, output_offset),
        [] => {
            let lower = input_offset.max(output_offset);
            let upper = (input_offset + input.len()).min(output_offset + output.len());
            let input_lower = lower - input_offset;
            let input_upper = upper - input_offset;
            let output_lower = lower - output_offset;
            let output_upper = upper - output_offset;

            let input_iter = iter!(input[input_lower..input_upper]);
            let output_iter = iter_mut!(output[output_lower..output_upper]);
            input_iter
                .zip(output_iter)
                .for_each(|(input, out)| *out = input.clone());
        }
        _ => {
            let mat_indices: Vec<usize> = ops
                .iter()
                .flat_map(|op| -> Vec<usize> {
                    (0..num_indices(op)).map(|i| get_index(op, i)).collect()
                })
                .collect();

            let row_fn = |(outputrow, outputloc): (usize, &mut P)| {
                let row = output_offset + (outputrow);
                let matrow = full_to_sub(n, &mat_indices, row);
                // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
                // for the row calculated above.
                let f = |(i, val): (usize, P)| -> P {
                    let colbits = sub_to_full(n, &mat_indices, i, row);
                    if colbits < input_offset {
                        P::zero()
                    } else {
                        let vecrow = colbits - input_offset;
                        if vecrow >= input.len() {
                            P::zero()
                        } else {
                            val * input[vecrow].clone()
                        }
                    }
                };

                // Get value for row and assign
                *outputloc = sum_for_ops_cols(matrow, ops, f);
            };

            // Generate output for each output row
            iter_mut!(output).enumerate().for_each(row_fn);
        }
    }
}
