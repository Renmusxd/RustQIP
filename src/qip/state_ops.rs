/// Contains functions, structs, and enums for storing and manipulating the quantum state.

extern crate num;
extern crate rayon;

use num::complex::Complex;
use rayon::prelude::*;

use QubitOp::*;
use crate::qubit_iterators::{ControlledOpIterator, SwapOpIterator};
use crate::utils::*;
use crate::qubit_iterators::MatrixOpIterator;
pub const PARALLEL_THRESHOLD: u64 = 12;

/// Types of unitary ops which can be applied to a state.
#[derive(Debug)]
pub enum QubitOp {
    // Indices, Matrix data
    MatrixOp(Vec<u64>, Vec<Complex<f64>>),
    // A indices, B indices
    SwapOp(Vec<u64>, Vec<u64>),
    // Control indices, Op indices, Op
    ControlOp(Vec<u64>, Vec<u64>, Box<QubitOp>),
}

/// Make a ControlOp
///
/// # Example
/// ```
/// use qip::state_ops::make_control_op;
/// use qip::state_ops::QubitOp::{MatrixOp, ControlOp};
/// let op = MatrixOp(vec![1], vec![/* ... */]);
/// let cop = make_control_op(vec![0], op);
///
/// if let ControlOp(c_indices, o_indices, _) = cop {
///     assert_eq!(c_indices, vec![0]);
///     assert_eq!(o_indices, vec![1]);
/// } else {
///     assert!(false);
/// }
/// ```
pub fn make_control_op(mut c_indices: Vec<u64>, op: QubitOp) -> QubitOp {
    match op {
        ControlOp(oc_indices, oo_indices, op) => {
            c_indices.extend(oc_indices);
            ControlOp(c_indices, oo_indices, op)
        }
        op => {
            let o_indices = (0..num_indices(&op)).map(|i| get_index(&op, i)).collect();
            ControlOp(c_indices, o_indices, Box::new(op))
        }
    }
}

/// Make a vector of complex numbers whose reals are given by `data`
pub fn from_reals(data: &[f64]) -> Vec<Complex<f64>> {
    data.into_iter().map(|x| Complex::<f64> {
        re: x.clone(),
        im: 0.0,
    }).collect()
}

/// Make a vector of complex numbers whose reals are given by the first tuple entry in `data` and
/// whose imaginaries are from the second.
pub fn from_tuples(data: &[(f64, f64)]) -> Vec<Complex<f64>> {
    data.into_iter().map(|x| -> Complex<f64> {
        let (r, i) = x;
        Complex::<f64> {
            re: r.clone(),
            im: i.clone(),
        }
    }).collect()
}

/// Get the number of indices represented by `op`
pub fn num_indices(op: &QubitOp) -> usize {
    match &op {
        MatrixOp(indices, _) => indices.len(),
        SwapOp(a, b) => a.len() + b.len(),
        ControlOp(cs, os, _) => cs.len() + os.len()
    }
}

/// Get the `i`th qubit index for `op`
pub fn get_index(op: &QubitOp, i: usize) -> u64 {
    match &op {
        MatrixOp(indices, _) => indices[i],
        SwapOp(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        ControlOp(cs, os, _) => {
            if i < cs.len() {
                cs[i]
            } else {
                os[i - cs.len()]
            }
        }
    }
}

/// Given the full matrix `row` and `col`, find the given op's row and column using the full `n`,
/// the op's `nindices`, the op's `indices'.
pub fn select_matrix_coords(n: u64, nindices: u64, indices: &Vec<u64>, row: u64, col: u64) -> (u64, u64) {
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

/// Get the row for a submatrix indexed by `indices` given the row for the larger 2^n by 2^n matrix.
pub fn select_matrix_row(n: u64, nindices: u64, indices: &Vec<u64>, row: u64) -> u64 {
    (0..nindices).fold(0, |x, j| -> u64 {
        let indx = indices[j as usize];
        let rowbit = get_bit(row, n - 1 - indx);
        let x = set_bit(x, nindices - 1 - j, rowbit);
        x
    })
}

/// Get the col for a submatrix indexed by `indices` given the col for the larger 2^n by 2^n matrix.
pub fn select_matrix_col(n: u64, nindices: u64, indices: &Vec<u64>, col: u64) -> u64 {
    (0..nindices).fold(0, |y, j| -> u64 {
        let indx = indices[j as usize];
        let colbit = get_bit(col, n - 1 - indx);
        let y = set_bit(y, nindices - 1 - j, colbit);
        y
    })
}

/// Builds a ControlledOpIterator for the given `op`, then maps using `f` and sums.
fn map_with_control_iterator<F: Fn((u64, Complex<f64>)) -> Complex<f64>>(nindices: u64, row: u64, op: &Box<QubitOp>, c_indices: &Vec<u64>, o_indices: &Vec<u64>, f: F) -> Complex<f64> {
    let n_control_indices = c_indices.len() as u64;
    let n_op_indices = o_indices.len() as u64;
    // Get reference to boxed op
    match &**op {
        MatrixOp(_, data) => {
            let iter_builder = |row: u64| MatrixOpIterator::new(row, n_op_indices, data);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.map(f).sum()
        }
        SwapOp(_, _) => {
            let iter_builder = |row: u64| SwapOpIterator::new(row, nindices);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.map(f).sum()
        }
        ControlOp(_, _, _) => {
            unimplemented!()
        }
    }
}

/// Using the function `f` which maps from a column and `row` to a complex value for the op matrix,
/// sums for all nonzero entries for a given `op` more efficiently than trying each column between
/// 0 and 2^nindices.
/// This really needs to be cleaned up, but runs in a tight loop. This makes it hard since Box
/// is unfeasible and the iterator types aren't the same size.
fn sum_for_op_cols<F: Fn((u64, Complex<f64>)) -> Complex<f64>>(nindices: u64, row: u64, op: &QubitOp, f: F) -> Complex<f64> {
    match op {
        MatrixOp(_,data) => {
            MatrixOpIterator::new(row, nindices, data).map(f).sum()
        },
        SwapOp(_, _) => {
            SwapOpIterator::new(row, nindices).map(f).sum()
        },
        ControlOp(c_indices, o_indices, op) => {
            map_with_control_iterator(nindices, row, op, c_indices, o_indices, f)
        }
    }
}

// TODO doc
pub fn apply_op(n: u64, op: &QubitOp,
               input: &Vec<Complex<f64>>, output: &mut Vec<Complex<f64>>,
               input_offset: u64, output_offset: u64, multi_core: bool) {
    let mat_indices: Vec<u64> = (0 .. num_indices(op)).map(|i| get_index(op, i)).collect();
    let mut flat_indices = mat_indices.clone();
    flat_indices.sort();
    let flat_indices = flat_indices;
    let nindices = mat_indices.len() as u64;

    let row_fn  = |(outputrow, outputloc): (usize, &mut Complex<f64>)| {
        let row = output_offset + (outputrow as u64);
        let matrow = select_matrix_row(n, nindices, &mat_indices, row);
        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
        // for the row calculated above.
        let f = |item: (u64, Complex<f64>)| -> Complex<f64> {
            let (i, val) = item;
            let colbits = (0..nindices).fold(row as u64, |acc, j| {
                let indx = flat_indices[j as usize];
                let bit_val = get_bit(i, nindices - 1 - j);
                set_bit(acc, n - 1 - indx, bit_val)
            });
            if colbits < input_offset {
                Complex::<f64> {
                    re: 0.0,
                    im: 0.0,
                }
            } else {
                let vecrow = colbits - input_offset;
                if vecrow >= input.len() as u64 {
                    Complex::<f64> {
                        re: 0.0,
                        im: 0.0,
                    }
                } else {
                    val * input[vecrow as usize]
                }
            }
        };

        // Get value for row and assign
        *outputloc = sum_for_op_cols(nindices, matrow, op, f);
    };

    // Generate output for each output row
    if multi_core {
        output.par_iter_mut().enumerate().for_each(row_fn);
    } else {
        output.iter_mut().enumerate().for_each(row_fn);
    }
}

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix(n: u64, op: &QubitOp) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n).map(|i| {
        let mut input = from_reals(&zeros);
        let mut output = input.clone();
        input[i] = Complex::<f64> {
            re: 1.0,
            im: 0.0,
        };
        apply_op(n, op, &input, &mut output, 0, 0, n > PARALLEL_THRESHOLD);
        output.clone()
    }).collect()
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
        let op = MatrixOp(vec![0, 1, 2], vec![]);
        assert_eq!(num_indices(&op), 3);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
    }

    #[test]
    fn test_get_index_condition() {
        let mop = MatrixOp(vec![2, 3], vec![]);
        let op = make_control_op(vec![0, 1], mop);
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_get_index_swap() {
        let op = SwapOp(vec![0, 1], vec![2, 3]);
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_apply_identity() {
        let op = MatrixOp(vec![0], from_reals(&vec![1.0, 0.0, 0.0, 1.0]));
        let input = from_reals(&vec![1.0, 0.0]);
        let mut output = from_reals(&vec![0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap() {
        let op = MatrixOp(vec![0], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&vec![1.0, 0.0]);
        let mut output = from_reals(&vec![0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_first() {
        let op = MatrixOp(vec![0], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let input = from_reals(&vec![1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&vec![0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = MatrixOp(vec![1], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let mut output = from_reals(&vec![0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(expected, output);
    }
}