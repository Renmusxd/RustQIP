use crate::iterators::{sum_for_ops_cols, MatrixOp};
use crate::utils::{get_bit, set_bit};
use crate::{iter, iter_mut};
use num_traits::{One, Zero};
use std::iter::Sum;
use std::ops::{AddAssign, Mul};

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
    op.indices()[i]
}

/// Compute vector element at a given row from op applied to input.
pub fn apply_op_row<P>(
    n: usize,
    op: &MatrixOp<P>,
    input: &[P],
    outputrow: usize,
    input_offset: usize,
    output_offset: usize,
) -> P
where
    P: Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    let mat_indices: Vec<usize> = (0..op.num_indices()).map(|i| get_index(op, i)).collect();
    apply_op_row_indices(
        n,
        op,
        input,
        outputrow,
        input_offset,
        output_offset,
        &mat_indices,
    )
}

/// Compute vector element at a given row from op applied to input.
pub fn apply_op_row_indices<P>(
    n: usize,
    op: &MatrixOp<P>,
    input: &[P],
    outputrow: usize,
    input_offset: usize,
    output_offset: usize,
    mat_indices: &[usize],
) -> P
where
    P: Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    let row = output_offset + (outputrow);
    let matrow = full_to_sub(n, mat_indices, row);
    // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
    // for the row calculated above.
    let f = |(i, val): (usize, P)| -> P {
        let colbits = sub_to_full(n, mat_indices, i, row);
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

    // Get value for row
    op.sum_for_op_cols(mat_indices.len(), matrow, f)
}

/// Apply `op` to the `input`, adding the results to `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
pub fn apply_op<P>(
    n: usize,
    op: &MatrixOp<P>,
    input: &[P],
    output: &mut [P],
    input_offset: usize,
    output_offset: usize,
) where
    P: AddAssign + Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    let mat_indices: Vec<usize> = (0..op.num_indices()).map(|i| get_index(op, i)).collect();
    let row_fn = |(outputrow, outputloc): (usize, &mut P)| {
        *outputloc += apply_op_row_indices(
            n,
            op,
            input,
            outputrow,
            input_offset,
            output_offset,
            &mat_indices,
        );
    };

    // Generate output for each output row
    iter_mut!(output).enumerate().for_each(row_fn);
}

/// Apply `op` to the `input`, adding the results to `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
pub fn apply_op_overwrite<P>(
    n: usize,
    op: &MatrixOp<P>,
    input: &[P],
    output: &mut [P],
    input_offset: usize,
    output_offset: usize,
) where
    P: AddAssign + Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
{
    let mat_indices: Vec<usize> = (0..op.num_indices()).map(|i| get_index(op, i)).collect();
    let row_fn = |(outputrow, outputloc): (usize, &mut P)| {
        *outputloc = apply_op_row_indices(
            n,
            op,
            input,
            outputrow,
            input_offset,
            output_offset,
            &mat_indices,
        );
    };

    // Generate output for each output row
    iter_mut!(output).enumerate().for_each(row_fn);
}

/// Apply `ops` to the `input`, adding the results to `output`. If either start at a nonzero state
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
    P: AddAssign + Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
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
                    (0..op.num_indices()).map(|i| get_index(op, i)).collect()
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
                *outputloc += sum_for_ops_cols(matrow, ops, f);
            };

            // Generate output for each output row
            iter_mut!(output).enumerate().for_each(row_fn);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array2, ShapeError};
    use std::ops::{Add, Div, Sub};

    /// Make the full op matrix from `ops`.
    /// Not very efficient, use only for debugging.
    fn make_op_matrix<P>(n: usize, op: &MatrixOp<P>) -> Vec<Vec<P>>
    where
        P: AddAssign + Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
    {
        let zeros: Vec<P> = (0..1 << n).map(|_| P::zero()).collect();
        (0..1 << n)
            .map(|i| {
                let mut input = zeros.clone();
                let mut output = zeros.clone();
                input[i] = P::one();
                apply_op(n, op, &input, &mut output, 0, 0);
                output
            })
            .collect()
    }

    fn make_op_flat_matrix<P>(n: usize, op: &MatrixOp<P>) -> Result<Array2<P>, ShapeError>
    where
        P: AddAssign + Clone + One + Zero + Sum + Mul<Output = P> + Send + Sync,
    {
        let v = make_op_matrix(n, op)
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect::<Vec<_>>();
        let arr = Array2::from_shape_vec((1 << n, 1 << n), v)?;
        Ok(arr.reversed_axes())
    }

    fn ndarray_kron_helper<P>(before: usize, mut mat: Array2<P>, after: usize) -> Array2<P>
    where
        P: Copy + One + Zero + Add<Output = P> + Sub<Output = P> + Div<Output = P> + 'static,
    {
        let eye = Array2::eye(2);
        for _ in 0..before {
            mat = ndarray::linalg::kron(&eye, &mat);
        }
        for _ in 0..after {
            mat = ndarray::linalg::kron(&mat, &eye);
        }
        mat
    }

    #[test]
    fn test_ident() -> Result<(), String> {
        let n = 3;
        let data = [1, 0, 0, 1];
        let op = MatrixOp::new_matrix([0], data);
        let arr = Array2::from_shape_vec((2, 2), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(0, arr, n - 1);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_flip() -> Result<(), String> {
        let n = 3;
        let data = [0, 1, 1, 0];
        let op = MatrixOp::new_matrix([0], data);
        let arr = Array2::from_shape_vec((2, 2), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(0, arr, n - 1);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_flip_mid() -> Result<(), String> {
        let n = 3;
        let data = [0, 1, 1, 0];
        let op = MatrixOp::new_matrix([1], data);
        let arr = Array2::from_shape_vec((2, 2), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(1, arr, n - 2);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_flip_end() -> Result<(), String> {
        let n = 3;
        let data = [0, 1, 1, 0];
        let op = MatrixOp::new_matrix([2], data);
        let arr = Array2::from_shape_vec((2, 2), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(2, arr, n - 3);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_flip_mid_twobody() -> Result<(), String> {
        let n = 4;
        // Hopping with number conservation
        let data = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1];
        let op = MatrixOp::new_matrix([1, 2], data);
        let arr = Array2::from_shape_vec((4, 4), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(1, arr, 1);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_counting() -> Result<(), String> {
        let n = 3;
        let data = [1, 2, 3, 4];
        let op = MatrixOp::new_matrix([0], data);
        let arr = Array2::from_shape_vec((2, 2), data.into()).map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;
        let comp_mat = ndarray_kron_helper(0, arr, n - 1);

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_counting_order() -> Result<(), String> {
        let n = 2;
        let data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let op = MatrixOp::new_matrix([0, 1], data);
        let comp_mat = Array2::from_shape_vec((1 << n, 1 << n), data.into())
            .map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;

        debug_assert_eq!(mat, comp_mat);
        Ok(())
    }

    #[test]
    fn test_counting_order_flipped() -> Result<(), String> {
        let n = 2;
        let data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let op = MatrixOp::new_matrix([1, 0], data);
        let comp_mat = Array2::from_shape_vec((1 << n, 1 << n), data.into())
            .map_err(|e| format!("{:?}", e))?;
        let mat = make_op_flat_matrix(n, &op).map_err(|e| format!("{:?}", e))?;

        debug_assert_ne!(mat, comp_mat);
        Ok(())
    }
}
