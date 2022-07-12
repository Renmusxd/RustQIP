/// Contains functions, structs, and enums for storing and manipulating the quantum state.
use crate::errors::{CircuitError, CircuitResult};
use crate::state_ops::iterators::*;
use crate::types::Representation;
use crate::utils::*;
use crate::{iter, iter_mut};
use crate::{Complex, Precision};
use num_traits::One;
use std::cmp::{max, min};

/// Make a Matrix UnitaryOp
pub fn make_matrix_op<P: Precision>(
    indices: Vec<usize>,
    dat: Vec<Complex<P>>,
) -> CircuitResult<UnitaryOp<P>> {
    let n = indices.len();
    let expected_mat_size = 1 << (2 * n);
    if indices.is_empty() {
        Err(CircuitError::new("Must supply at least one op index"))
    } else if dat.len() != expected_mat_size {
        let message = format!(
            "Matrix data has {:?} entries versus expected 2^2*{:?}",
            dat.len(),
            n
        );
        Err(CircuitError::new(message))
    } else {
        Ok(UnitaryOp::<P>::Matrix(indices, dat))
    }
}

/// Make a SparseMatrix UnitaryOp from a vector of rows (with `(column, value)`).
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `column`
/// and `row` where `row` is the index of `dat`.
pub fn make_sparse_matrix_op<P: Precision>(
    indices: Vec<usize>,
    dat: Vec<Vec<(usize, Complex<P>)>>,
    order: Representation,
) -> CircuitResult<UnitaryOp<P>> {
    let n = indices.len();
    let expected_mat_size = 1 << n;
    if indices.is_empty() {
        Err(CircuitError::new("Must supply at least one op index"))
    } else if dat.len() != expected_mat_size {
        let message = format!(
            "Sparse matrix has {:?} rows versus expected 2^{:?}",
            dat.len(),
            n
        );
        Err(CircuitError::new(message))
    } else {
        // Each row needs at least one entry
        dat.iter().enumerate().try_for_each(|(row, v)| {
            if v.is_empty() {
                let message = format!(
                    "All rows of sparse matrix must have data ({:?} is empty)",
                    row
                );
                Err(CircuitError::new(message))
            } else {
                Ok(())
            }
        })?;

        let dat = match order {
            Representation::LittleEndian => {
                let mut dat: Vec<_> = dat
                    .into_iter()
                    .map(|v| {
                        v.into_iter()
                            .map(|(indx, c)| (flip_bits(n, indx), c))
                            .collect()
                    })
                    .enumerate()
                    .collect();
                dat.sort_by_key(|(indx, _)| flip_bits(n, *indx));
                dat.into_iter().map(|(_, c)| c).collect()
            }
            Representation::BigEndian => dat,
        };

        Ok(UnitaryOp::<P>::SparseMatrix(indices, dat))
    }
}

/// Make a vector of vectors of rows (with `(column, value)`) built from a function
/// `f` which takes row numbers.
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `row` and
/// the output `column` from `f`
pub fn make_sparse_matrix_from_function<P: Precision, F: Fn(usize) -> Vec<(usize, Complex<P>)>>(
    n: usize,
    f: F,
    order: Representation,
) -> Vec<Vec<(usize, Complex<P>)>> {
    (0..1 << n)
        .map(|indx| {
            let indx = match order {
                Representation::LittleEndian => flip_bits(n, indx),
                Representation::BigEndian => indx,
            };
            let v = f(indx);
            match order {
                Representation::LittleEndian => v
                    .into_iter()
                    .map(|(indx, c)| (flip_bits(n, indx), c))
                    .collect(),
                Representation::BigEndian => v,
            }
        })
        .collect()
}

/// Make a Swap UnitaryOp
pub fn make_swap_op<P: Precision>(
    a_indices: Vec<usize>,
    b_indices: Vec<usize>,
) -> CircuitResult<UnitaryOp<P>> {
    if a_indices.is_empty() || b_indices.is_empty() {
        Err(CircuitError::new("Need at least 1 swap index for a and b"))
    } else if a_indices.len() != b_indices.len() {
        let message = format!(
            "Swap must be performed on two sets of indices of equal length, found {:?} vs {:?}",
            a_indices.len(),
            b_indices.len()
        );
        Err(CircuitError::new(message))
    } else {
        Ok(UnitaryOp::Swap(a_indices, b_indices))
    }
}

/// Make a Control UnitaryOp
pub fn make_control_op<P: Precision>(
    mut c_indices: Vec<usize>,
    op: UnitaryOp<P>,
) -> CircuitResult<UnitaryOp<P>> {
    if c_indices.is_empty() {
        Err(CircuitError::new("Must supply at least one control index"))
    } else {
        match op {
            UnitaryOp::Control(oc_indices, oo_indices, op) => {
                c_indices.extend(oc_indices);
                Ok(UnitaryOp::Control(c_indices, oo_indices, op))
            }
            op => {
                let o_indices = (0..num_indices(&op)).map(|i| get_index(&op, i)).collect();
                Ok(UnitaryOp::Control(c_indices, o_indices, Box::new(op)))
            }
        }
    }
}

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn invert_op<P: Precision>(op: UnitaryOp<P>) -> UnitaryOp<P> {
    conj_op(transpose_op(op))
}

/// Get conjugate of op.
pub fn conj_op<P: Precision>(op: UnitaryOp<P>) -> UnitaryOp<P> {
    match op {
        UnitaryOp::Matrix(indices, mat) => {
            let mat = mat.into_iter().map(|v| v.conj()).collect();
            UnitaryOp::Matrix(indices, mat)
        }
        UnitaryOp::SparseMatrix(indices, mat) => {
            let mat = mat
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|(indx, val)| (indx, val.conj()))
                        .collect()
                })
                .collect();
            UnitaryOp::SparseMatrix(indices, mat)
        }
        UnitaryOp::Swap(a_indices, b_indices) => UnitaryOp::Swap(a_indices, b_indices),
        UnitaryOp::Control(c_indices, op_indices, op) => {
            UnitaryOp::Control(c_indices, op_indices, Box::new(conj_op(*op)))
        }
    }
}

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn transpose_op<P: Precision>(op: UnitaryOp<P>) -> UnitaryOp<P> {
    match op {
        UnitaryOp::Matrix(indices, mut mat) => {
            let n = indices.len();
            (0..1 << n).for_each(|row| {
                (0..row).for_each(|col| {
                    mat.swap(get_flat_index(n, row, col), get_flat_index(n, col, row));
                })
            });
            UnitaryOp::Matrix(indices, mat)
        }
        UnitaryOp::SparseMatrix(indices, mat) => {
            UnitaryOp::SparseMatrix(indices, transpose_sparse(mat))
        }
        UnitaryOp::Swap(a_indices, b_indices) => UnitaryOp::Swap(a_indices, b_indices),
        UnitaryOp::Control(c_indices, op_indices, op) => {
            UnitaryOp::Control(c_indices, op_indices, Box::new(transpose_op(*op)))
        }
    }
}

/// Make a vector of complex numbers whose reals are given by `data`
pub fn from_reals<P: Precision>(data: &[P]) -> Vec<Complex<P>> {
    data.iter()
        .map(|x| Complex::<P> {
            re: *x,
            im: P::zero(),
        })
        .collect()
}

/// Make a vector of complex numbers whose reals are given by the first tuple entry in `data` and
/// whose imaginaries are from the second.
pub fn from_tuples<P: Precision>(data: &[(P, P)]) -> Vec<Complex<P>> {
    data.iter()
        .map(|x| -> Complex<P> {
            let (r, i) = x;
            Complex::<P> { re: *r, im: *i }
        })
        .collect()
}

/// Given the full matrix `row` and `col`, find the given op's row and column using the full `n`,
/// the op's `nindices`, the op's `indices'.
pub fn select_matrix_coords(
    n: usize,
    nindices: usize,
    indices: &[usize],
    row: usize,
    col: usize,
) -> (usize, usize) {
    (0..nindices).fold((0, 0), |acc, j| -> (usize, usize) {
        let (x, y) = acc;
        let indx = indices[j];
        let rowbit = get_bit(row, n - 1 - indx);
        let colbit = get_bit(col, n - 1 - indx);
        let x = set_bit(x, nindices - 1 - j, rowbit);
        let y = set_bit(y, nindices - 1 - j, colbit);
        (x, y)
    })
}

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

/// Get the number of indices represented by `op`
pub fn num_indices<P: Precision>(op: &UnitaryOp<P>) -> usize {
    match &op {
        UnitaryOp::Matrix(indices, _) => indices.len(),
        UnitaryOp::SparseMatrix(indices, _) => indices.len(),
        UnitaryOp::Swap(a, b) => a.len() + b.len(),
        UnitaryOp::Control(cs, os, _) => cs.len() + os.len(),
    }
}

/// Get the `i`th qubit index for `op`
pub fn get_index<P: Precision>(op: &UnitaryOp<P>, i: usize) -> usize {
    match &op {
        UnitaryOp::Matrix(indices, _) => indices[i],
        UnitaryOp::SparseMatrix(indices, _) => indices[i],
        UnitaryOp::Swap(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        UnitaryOp::Control(cs, os, _) => {
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
pub fn apply_op<P: Precision>(
    n: usize,
    op: &UnitaryOp<P>,
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    input_offset: usize,
    output_offset: usize,
) {
    let mat_indices: Vec<usize> = (0..num_indices(op)).map(|i| get_index(op, i)).collect();
    let nindices = mat_indices.len();

    let row_fn = |(outputrow, outputloc): (usize, &mut Complex<P>)| {
        let row = output_offset + (outputrow);
        let matrow = full_to_sub(n, &mat_indices, row);
        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
        // for the row calculated above.
        let f = |(i, val): (usize, Complex<P>)| -> Complex<P> {
            let colbits = sub_to_full(n, &mat_indices, i, row);
            if colbits < input_offset {
                Complex::default()
            } else {
                let vecrow = colbits - input_offset;
                if vecrow >= input.len() {
                    Complex::default()
                } else {
                    val * input[vecrow]
                }
            }
        };

        // Get value for row and assign
        *outputloc = sum_for_op_cols(nindices, matrow, op, f);
    };

    // Generate output for each output row
    iter_mut!(output).enumerate().for_each(row_fn);
}

/// Apply `ops` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
/// This is much less efficient as compared to repeated applications of `apply_op`, if your ops can
/// be applied in sequence, do so with `apply_op`.
pub fn apply_ops<P: Precision>(
    n: usize,
    ops: &[UnitaryOp<P>],
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    input_offset: usize,
    output_offset: usize,
) {
    match ops {
        [op] => apply_op(n, op, input, output, input_offset, output_offset),
        [] => {
            let lower = max(input_offset, output_offset);
            let upper = min(input_offset + input.len(), output_offset + output.len());
            let input_lower = lower - input_offset;
            let input_upper = upper - input_offset;
            let output_lower = lower - output_offset;
            let output_upper = upper - output_offset;

            let input_iter = iter!(input[input_lower..input_upper]);
            let output_iter = iter_mut!(output[output_lower..output_upper]);
            input_iter
                .zip(output_iter)
                .for_each(|(input, out)| *out = *input);
        }
        _ => {
            let mat_indices: Vec<usize> = ops
                .iter()
                .flat_map(|op| -> Vec<usize> {
                    (0..num_indices(op)).map(|i| get_index(op, i)).collect()
                })
                .collect();

            let row_fn = |(outputrow, outputloc): (usize, &mut Complex<P>)| {
                let row = output_offset + (outputrow);
                let matrow = full_to_sub(n, &mat_indices, row);
                // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
                // for the row calculated above.
                let f = |(i, val): (usize, Complex<P>)| -> Complex<P> {
                    let colbits = sub_to_full(n, &mat_indices, i, row);
                    if colbits < input_offset {
                        Complex::default()
                    } else {
                        let vecrow = colbits - input_offset;
                        if vecrow >= input.len() {
                            Complex::default()
                        } else {
                            val * input[vecrow]
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

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix<P: Precision>(n: usize, op: &UnitaryOp<P>) -> Vec<Vec<Complex<P>>> {
    let zeros: Vec<P> = (0..1 << n).map(|_| P::zero()).collect();
    (0..1 << n)
        .map(|i| {
            let mut input = from_reals(&zeros);
            let mut output = input.clone();
            input[i] = Complex::one();
            apply_op(n, op, &input, &mut output, 0, 0);
            output
        })
        .collect()
}

#[cfg(test)]
mod state_ops_tests {
    use super::*;

    #[test]
    fn test_get_bit() {
        assert!(!get_bit(1, 1));
        assert!(get_bit(1, 0));
    }

    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(1, 0, true), 1);
        assert_eq!(set_bit(1, 1, true), 3);
    }

    #[test]
    fn test_get_index_simple() {
        let op = UnitaryOp::<f64>::Matrix(vec![0, 1, 2], vec![]);
        assert_eq!(num_indices(&op), 3);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
    }

    #[test]
    fn test_get_index_condition() {
        let mop = UnitaryOp::<f64>::Matrix(vec![2, 3], vec![]);
        let op = make_control_op(vec![0, 1], mop).unwrap();
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_get_index_swap() {
        let op = UnitaryOp::<f64>::Swap(vec![0, 1], vec![2, 3]);
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_apply_identity() {
        let op = UnitaryOp::<f64>::Matrix(vec![0], from_reals(&[1.0, 0.0, 0.0, 1.0]));
        let input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat() {
        let op = UnitaryOp::<f64>::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat_first() {
        let op = UnitaryOp::<f64>::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));

        let input = from_reals(&[1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0);

        let expected = from_reals(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = UnitaryOp::<f64>::Matrix(vec![1], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0);

        let expected = from_reals(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(expected, output);
    }

    #[test]
    fn test_make_sparse_mat() {
        let one = Complex::<f64>::one();
        let expected_dat = vec![
            vec![(1, one)],
            vec![(0, one)],
            vec![(3, one)],
            vec![(2, one)],
        ];
        let op1 =
            make_sparse_matrix_op(vec![0, 1], expected_dat.clone(), Representation::BigEndian)
                .unwrap();
        let op2 = make_sparse_matrix_op(
            vec![0, 1],
            vec![
                vec![(2, one)],
                vec![(3, one)],
                vec![(0, one)],
                vec![(1, one)],
            ],
            Representation::LittleEndian,
        )
        .unwrap();

        // Both should not be in natural order.
        if let UnitaryOp::SparseMatrix(_, data) = op1 {
            assert_eq!(data, expected_dat);
        }
        if let UnitaryOp::SparseMatrix(_, data) = op2 {
            assert_eq!(data, expected_dat);
        }
    }
}
