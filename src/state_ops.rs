/// Contains functions, structs, and enums for storing and manipulating the quantum state.
extern crate rayon;
use rayon::prelude::*;

use crate::errors::CircuitError;
use crate::iterators::*;
use crate::utils::*;
use crate::{Complex, Precision};
use num::{One, Zero};
use std::cmp::{max, min};
use std::fmt;

/// Types of unitary ops which can be applied to a state.
pub enum UnitaryOp {
    /// Indices, Matrix data
    Matrix(Vec<u64>, Vec<Complex<f64>>),
    /// Indices, Matrix data
    SparseMatrix(Vec<u64>, Vec<Vec<(u64, Complex<f64>)>>),
    /// A indices, B indices
    Swap(Vec<u64>, Vec<u64>),
    /// Control indices, Op indices, Op
    Control(Vec<u64>, Vec<u64>, Box<UnitaryOp>),
    /// Function which maps |x,y> to |x,f(x) xor y>
    Function(
        Vec<u64>,
        Vec<u64>,
        Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ),
}

/// Cannot clone functions, so converts them to sparse matrices.
impl Clone for UnitaryOp {
    fn clone(&self) -> Self {
        match self {
            UnitaryOp::Matrix(indices, mat) => UnitaryOp::Matrix(indices.clone(), mat.clone()),
            UnitaryOp::SparseMatrix(indices, mat) => {
                let mat = mat.to_vec();
                UnitaryOp::SparseMatrix(indices.clone(), mat)
            }
            UnitaryOp::Swap(a_indices, b_indices) => {
                UnitaryOp::Swap(a_indices.clone(), b_indices.clone())
            }
            UnitaryOp::Control(c_indices, op_indices, op) => {
                UnitaryOp::Control(c_indices.clone(), op_indices.clone(), op.clone())
            }
            UnitaryOp::Function(x_indices, y_indices, f) => {
                let n = (x_indices.len() + y_indices.len()) as u64;
                let mat = (0..1 << n)
                    .map(|col| {
                        let (row, phase) = f(col);
                        let val = Complex { re: 0.0, im: phase };
                        vec![(row, val.exp())]
                    })
                    .collect();
                let indices = x_indices.iter().chain(y_indices.iter()).cloned().collect();
                invert_op(UnitaryOp::SparseMatrix(indices, mat))
            }
        }
    }
}

impl fmt::Debug for UnitaryOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (name, indices) = match self {
            UnitaryOp::Matrix(indices, _) => ("Matrix".to_string(), indices.clone()),
            UnitaryOp::SparseMatrix(indices, _) => ("SparseMatrix".to_string(), indices.clone()),
            UnitaryOp::Swap(a_indices, b_indices) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("Swap".to_string(), indices)
            }
            UnitaryOp::Control(indices, _, op) => {
                let name = format!("C({:?})", *op);
                (name, indices.clone())
            }
            UnitaryOp::Function(a_indices, b_indices, _) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("F".to_string(), indices)
            }
        };
        let int_strings = indices
            .iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "{}[{}]", name, int_strings.join(", "))
    }
}

/// Make a Matrix UnitaryOp
pub fn make_matrix_op(
    indices: Vec<u64>,
    dat: Vec<Complex<f64>>,
) -> Result<UnitaryOp, CircuitError> {
    let n = indices.len();
    let expected_mat_size = 1 << (2 * n);
    if indices.is_empty() {
        CircuitError::make_str_err("Must supply at least one op index")
    } else if dat.len() != expected_mat_size {
        let message = format!(
            "Matrix data has {:?} entries versus expected 2^2*{:?}",
            dat.len(),
            n
        );
        CircuitError::make_err(message)
    } else {
        Ok(UnitaryOp::Matrix(indices, dat))
    }
}

/// Make a SparseMatrix UnitaryOp from a vector of rows (with `(column, value)`).
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `column`
/// and `row` where `row` is the index of `dat`.
pub fn make_sparse_matrix_op(
    indices: Vec<u64>,
    dat: Vec<Vec<(u64, Complex<f64>)>>,
    natural_order: bool,
) -> Result<UnitaryOp, CircuitError> {
    let n = indices.len();
    let expected_mat_size = 1 << n;
    if indices.is_empty() {
        CircuitError::make_str_err("Must supply at least one op index")
    } else if dat.len() != expected_mat_size {
        let message = format!(
            "Sparse matrix has {:?} rows versus expected 2^{:?}",
            dat.len(),
            n
        );
        CircuitError::make_err(message)
    } else {
        // Each row needs at least one entry
        dat.iter().enumerate().try_for_each(|(row, v)| {
            if v.is_empty() {
                let message = format!(
                    "All rows of sparse matrix must have data ({:?} is empty)",
                    row
                );
                CircuitError::make_err(message)
            } else {
                Ok(())
            }
        })?;

        let dat = if natural_order {
            let mut dat: Vec<_> = dat
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|(indx, c)| (flip_bits(n, indx), c))
                        .collect()
                })
                .enumerate()
                .collect();
            dat.sort_by_key(|(indx, _)| flip_bits(n, *indx as u64));
            dat.into_iter().map(|(_, c)| c).collect()
        } else {
            dat
        };

        Ok(UnitaryOp::SparseMatrix(indices, dat))
    }
}

/// Make a vector of vectors of rows (with `(column, value)`) built from a function
/// `f` which takes row numbers.
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `row` and
/// the output `column` from `f`
pub fn make_sparse_matrix_from_function<F: Fn(u64) -> Vec<(u64, Complex<f64>)>>(
    n: usize,
    f: F,
    natural_order: bool,
) -> Vec<Vec<(u64, Complex<f64>)>> {
    (0..1 << n as u64)
        .map(|indx| {
            let indx = if natural_order {
                flip_bits(n, indx)
            } else {
                indx
            };
            let v = f(indx);
            if natural_order {
                v.into_iter()
                    .map(|(indx, c)| (flip_bits(n, indx), c))
                    .collect()
            } else {
                v
            }
        })
        .collect()
}

/// Make a Swap UnitaryOp
pub fn make_swap_op(a_indices: Vec<u64>, b_indices: Vec<u64>) -> Result<UnitaryOp, CircuitError> {
    if a_indices.is_empty() || b_indices.is_empty() {
        CircuitError::make_str_err("Need at least 1 swap index for a and b")
    } else if a_indices.len() != b_indices.len() {
        let message = format!(
            "Swap must be performed on two sets of indices of equal length, found {:?} vs {:?}",
            a_indices.len(),
            b_indices.len()
        );
        CircuitError::make_err(message)
    } else {
        Ok(UnitaryOp::Swap(a_indices, b_indices))
    }
}

/// Make a Control UnitaryOp
///
/// # Example
/// ```
/// use qip::state_ops::make_control_op;
/// use qip::state_ops::UnitaryOp::{Matrix, Control};
/// let op = Matrix(vec![1], vec![/* ... */]);
/// let cop = make_control_op(vec![0], op).unwrap();
///
/// if let Control(c_indices, o_indices, _) = cop {
///     assert_eq!(c_indices, vec![0]);
///     assert_eq!(o_indices, vec![1]);
/// } else {
///     assert!(false);
/// }
/// ```
pub fn make_control_op(mut c_indices: Vec<u64>, op: UnitaryOp) -> Result<UnitaryOp, CircuitError> {
    if c_indices.is_empty() {
        CircuitError::make_str_err("Must supply at least one control index")
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

/// Make a Function UnitaryOp
pub fn make_function_op(
    input_indices: Vec<u64>,
    output_indices: Vec<u64>,
    f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
) -> Result<UnitaryOp, CircuitError> {
    if input_indices.is_empty() || output_indices.is_empty() {
        CircuitError::make_str_err("Input and Output indices must not be empty")
    } else {
        Ok(UnitaryOp::Function(input_indices, output_indices, f))
    }
}

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn invert_op(op: UnitaryOp) -> UnitaryOp {
    conj_op(transpose_op(op))
}

/// Get conjugate of op.
pub fn conj_op(op: UnitaryOp) -> UnitaryOp {
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
        UnitaryOp::Function(x_indices, y_indices, f) => {
            let n = (x_indices.len() + y_indices.len()) as u64;
            let mat = (0..1 << n)
                .map(|col| {
                    let (row, phase) = f(col);
                    let val = Complex {
                        re: 0.0,
                        im: -phase,
                    }
                    .exp();
                    vec![(row, val)]
                })
                .collect();
            let indices = x_indices.into_iter().chain(y_indices.into_iter()).collect();
            invert_op(UnitaryOp::SparseMatrix(indices, mat))
        }
    }
}

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn transpose_op(op: UnitaryOp) -> UnitaryOp {
    match op {
        UnitaryOp::Matrix(indices, mut mat) => {
            let n = indices.len() as u64;
            (0..1 << n).for_each(|row| {
                (0..row).for_each(|col| {
                    mat.swap(
                        get_flat_index(n, row, col) as usize,
                        get_flat_index(n, col, row) as usize,
                    );
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
        UnitaryOp::Function(x_indices, y_indices, f) => {
            let n = (x_indices.len() + y_indices.len()) as u64;
            let mat = (0..1 << n)
                .map(|col| {
                    let (row, phase) = f(col);
                    let val = Complex { re: 0.0, im: phase }.exp();
                    vec![(row, val)]
                })
                .collect();
            let indices = x_indices.into_iter().chain(y_indices.into_iter()).collect();
            invert_op(UnitaryOp::SparseMatrix(indices, mat))
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
    n: u64,
    nindices: u64,
    indices: &[u64],
    row: u64,
    col: u64,
) -> (u64, u64) {
    (0..nindices).fold((0, 0), |acc, j| -> (u64, u64) {
        let (x, y) = acc;
        let indx = indices[j as usize];
        let rowbit = get_bit(row, n - 1 - indx);
        let colbit = get_bit(col, n - 1 - indx);
        let x = set_bit(x, nindices - 1 - j, rowbit);
        let y = set_bit(y, nindices - 1 - j, colbit);
        (x, y)
    })
}

/// Get the index for a submatrix indexed by `indices` given the `full_index` for the larger 2^n by 2^n matrix.
pub fn full_to_sub(n: u64, mat_indices: &[u64], full_index: u64) -> u64 {
    let nindices = mat_indices.len() as u64;
    mat_indices
        .iter()
        .enumerate()
        .fold(0, |acc, (j, indx)| -> u64 {
            let bit = get_bit(full_index, n - 1 - *indx);
            set_bit(acc, nindices - 1 - j as u64, bit)
        })
}

/// Given the `sub_index` for the submatrix, and a base to overwrite values, get the full index for the 2^n by 2^n matrix.
pub fn sub_to_full(n: u64, mat_indices: &[u64], sub_index: u64, base: u64) -> u64 {
    let nindices = mat_indices.len() as u64;
    mat_indices.iter().enumerate().fold(base, |acc, (j, indx)| {
        let bit = get_bit(sub_index, nindices - 1 - j as u64);
        set_bit(acc, n - 1 - *indx, bit)
    })
}

/// Get the number of indices represented by `op`
pub fn num_indices(op: &UnitaryOp) -> usize {
    match &op {
        UnitaryOp::Matrix(indices, _) => indices.len(),
        UnitaryOp::SparseMatrix(indices, _) => indices.len(),
        UnitaryOp::Swap(a, b) => a.len() + b.len(),
        UnitaryOp::Control(cs, os, _) => cs.len() + os.len(),
        UnitaryOp::Function(inputs, outputs, _) => inputs.len() + outputs.len(),
    }
}

/// Get the `i`th qubit index for `op`
pub fn get_index(op: &UnitaryOp, i: usize) -> u64 {
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
        UnitaryOp::Function(inputs, outputs, _) => {
            if i < inputs.len() {
                inputs[i]
            } else {
                outputs[i - inputs.len()]
            }
        }
    }
}

/// Convert &UnitaryOp to equivalent PrecisionUnitaryOp<P>
pub(crate) fn clone_as_precision_op<P: Precision>(op: &UnitaryOp) -> PrecisionUnitaryOp<P> {
    match op {
        UnitaryOp::Matrix(indices, data) => {
            let data: Vec<_> = data
                .iter()
                .map(|c| Complex {
                    re: P::from(c.re).unwrap(),
                    im: P::from(c.im).unwrap(),
                })
                .collect();
            PrecisionUnitaryOp::Matrix(indices.clone(), data)
        }
        UnitaryOp::SparseMatrix(indices, data) => {
            let data: Vec<Vec<_>> = data
                .iter()
                .map(|v| {
                    v.iter()
                        .map(|(col, c)| {
                            (
                                *col,
                                Complex {
                                    re: P::from(c.re).unwrap(),
                                    im: P::from(c.im).unwrap(),
                                },
                            )
                        })
                        .collect()
                })
                .collect();
            PrecisionUnitaryOp::SparseMatrix(indices.clone(), data)
        }
        UnitaryOp::Swap(a_indices, b_indices) => {
            PrecisionUnitaryOp::Swap(a_indices.clone(), b_indices.clone())
        }
        UnitaryOp::Control(c_indices, o_indices, op) => PrecisionUnitaryOp::Control(
            c_indices.clone(),
            o_indices.clone(),
            Box::new(clone_as_precision_op(op)),
        ),
        UnitaryOp::Function(inputs, outputs, f) => {
            PrecisionUnitaryOp::Function(inputs.clone(), outputs.clone(), f)
        }
    }
}

/// Apply `op` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
pub fn apply_op<P: Precision>(
    n: u64,
    op: &UnitaryOp,
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    input_offset: u64,
    output_offset: u64,
    multithread: bool,
) {
    let op = clone_as_precision_op::<P>(op);
    let mat_indices: Vec<u64> = (0..precision_num_indices(&op))
        .map(|i| precision_get_index(&op, i))
        .collect();
    let nindices = mat_indices.len() as u64;

    let row_fn = |(outputrow, outputloc): (usize, &mut Complex<P>)| {
        let row = output_offset + (outputrow as u64);
        let matrow = full_to_sub(n, &mat_indices, row);
        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
        // for the row calculated above.
        let f = |(i, val): (u64, Complex<P>)| -> Complex<P> {
            let colbits = sub_to_full(n, &mat_indices, i, row);
            if colbits < input_offset {
                Complex::zero()
            } else {
                let vecrow = colbits - input_offset;
                if vecrow >= input.len() as u64 {
                    Complex::zero()
                } else {
                    val * input[vecrow as usize]
                }
            }
        };

        // Get value for row and assign
        *outputloc = sum_for_op_cols(nindices, matrow, &op, f);
    };

    // Generate output for each output row
    if multithread {
        output.par_iter_mut().enumerate().for_each(row_fn);
    } else {
        output.iter_mut().enumerate().for_each(row_fn);
    }
}

/// Apply `ops` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
/// This is much less efficient as compared to repeated applications of `apply_op`, if your ops can
/// be applied in sequence, do so with `apply_op`.
pub fn apply_ops<P: Precision>(
    n: u64,
    ops: &[&UnitaryOp],
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    input_offset: u64,
    output_offset: u64,
    multithread: bool,
) {
    match ops {
        [op] => apply_op(
            n,
            op,
            input,
            output,
            input_offset,
            output_offset,
            multithread,
        ),
        [] => {
            let lower = max(input_offset, output_offset);
            let upper = min(
                input_offset + input.len() as u64,
                output_offset + output.len() as u64,
            );
            let input_lower = (lower - input_offset) as usize;
            let input_upper = (upper - input_offset) as usize;
            let output_lower = (lower - output_offset) as usize;
            let output_upper = (upper - output_offset) as usize;

            if multithread {
                let input_iter = input[input_lower..input_upper].par_iter();
                let output_iter = output[output_lower..output_upper].par_iter_mut();
                input_iter
                    .zip(output_iter)
                    .for_each(|(input, out)| *out = *input);
            } else {
                let input_iter = input[input_lower..input_upper].iter();
                let output_iter = output[output_lower..output_upper].iter_mut();
                input_iter
                    .zip(output_iter)
                    .for_each(|(input, out)| *out = *input);
            }
        }
        _ => {
            let ops: Vec<_> = ops
                .iter()
                .map(|op| clone_as_precision_op::<P>(op))
                .collect();

            let mat_indices: Vec<u64> = ops
                .iter()
                .map(|op| -> Vec<u64> {
                    (0..precision_num_indices(&op))
                        .map(|i| precision_get_index(&op, i))
                        .collect()
                })
                .flatten()
                .collect();

            let row_fn = |(outputrow, outputloc): (usize, &mut Complex<P>)| {
                let row = output_offset + (outputrow as u64);
                let matrow = full_to_sub(n, &mat_indices, row);
                // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
                // for the row calculated above.
                let f = |(i, val): (u64, Complex<P>)| -> Complex<P> {
                    let colbits = sub_to_full(n, &mat_indices, i, row);
                    if colbits < input_offset {
                        Complex::zero()
                    } else {
                        let vecrow = colbits - input_offset;
                        if vecrow >= input.len() as u64 {
                            Complex::zero()
                        } else {
                            val * input[vecrow as usize]
                        }
                    }
                };

                // Get value for row and assign
                *outputloc = sum_for_ops_cols(matrow, &ops, f);
            };

            // Generate output for each output row
            if multithread {
                output.par_iter_mut().enumerate().for_each(row_fn);
            } else {
                output.iter_mut().enumerate().for_each(row_fn);
            }
        }
    }
}

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix<P: Precision>(
    n: u64,
    op: &UnitaryOp,
    multithread: bool,
) -> Vec<Vec<Complex<P>>> {
    let zeros: Vec<P> = (0..1 << n).map(|_| P::zero()).collect();
    (0..1 << n)
        .map(|i| {
            let mut input = from_reals(&zeros);
            let mut output = input.clone();
            input[i] = Complex::one();
            apply_op(n, op, &input, &mut output, 0, 0, multithread);
            output
        })
        .collect()
}

#[cfg(test)]
mod state_ops_tests {
    use super::*;

    #[test]
    fn test_get_bit() {
        assert_eq!(get_bit(1, 1), false);
        assert_eq!(get_bit(1, 0), true);
    }

    #[test]
    fn test_set_bit() {
        assert_eq!(set_bit(1, 0, true), 1);
        assert_eq!(set_bit(1, 1, true), 3);
    }

    #[test]
    fn test_get_index_simple() {
        let op = UnitaryOp::Matrix(vec![0, 1, 2], vec![]);
        assert_eq!(num_indices(&op), 3);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
    }

    #[test]
    fn test_get_index_condition() {
        let mop = UnitaryOp::Matrix(vec![2, 3], vec![]);
        let op = make_control_op(vec![0, 1], mop).unwrap();
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_get_index_swap() {
        let op = UnitaryOp::Swap(vec![0, 1], vec![2, 3]);
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_apply_identity() {
        let op = UnitaryOp::Matrix(vec![0], from_reals(&[1.0, 0.0, 0.0, 1.0]));
        let input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat() {
        let op = UnitaryOp::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat_first() {
        let op = UnitaryOp::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));

        let input = from_reals(&[1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = UnitaryOp::Matrix(vec![1], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(expected, output);
    }

    #[test]
    fn test_many_mat_swap() {
        let n = 5;

        let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
        let ops: Vec<_> = (0..n)
            .map(|indx| UnitaryOp::Matrix(vec![indx], mat.clone()))
            .collect();
        let r_ops: Vec<_> = ops.iter().collect();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        apply_ops(n, &r_ops, &input, &mut output, 0, 0, false);
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
        let op1 = make_sparse_matrix_op(vec![0, 1], expected_dat.clone(), false).unwrap();
        let op2 = make_sparse_matrix_op(
            vec![0, 1],
            vec![
                vec![(2, one)],
                vec![(3, one)],
                vec![(0, one)],
                vec![(1, one)],
            ],
            true,
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
