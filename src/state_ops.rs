/// Contains functions, structs, and enums for storing and manipulating the quantum state.
extern crate num;
extern crate rayon;

use num::complex::Complex;
use rayon::prelude::*;

use PrecisionQubitOp::*;

use crate::iterators::*;
use crate::types::Precision;
use crate::utils::*;

/// Types of unitary ops which can be applied to a state.
pub enum QubitOp {
    // Indices, Matrix data
    Matrix(Vec<u64>, Vec<Complex<f64>>),
    // A indices, B indices
    Swap(Vec<u64>, Vec<u64>),
    // Control indices, Op indices, Op
    Control(Vec<u64>, Vec<u64>, Box<QubitOp>),
    // Function which maps |x,y> to |x,f(x) xor y>
    Function(Vec<u64>, Vec<u64>, Box<Fn(u64) -> (u64, f64) + Send + Sync>),
}

/// Make a Matrix QubitOp
pub fn make_matrix_op(indices: Vec<u64>, dat: Vec<Complex<f64>>) -> Result<QubitOp, &'static str> {
    if indices.is_empty() {
        Err("Must supply at least one op index")
    } else {
        let expected_mat_size = 1 << (2 * indices.len());
        if dat.len() != expected_mat_size {
            Err("Matrix data not correct size for 2^n by 2^n matrix")
        } else {
            Ok(QubitOp::Matrix(indices, dat))
        }
    }
}

/// Make a Swap QubitOp
pub fn make_swap_op(a_indices: Vec<u64>, b_indices: Vec<u64>) -> Result<QubitOp, &'static str> {
    if a_indices.is_empty() || b_indices.is_empty() {
        Err("Need at least 1 swap index for a and b")
    } else if a_indices.len() != b_indices.len() {
        Err("Swap must be performed on two sets of indices of equal length")
    } else {
        Ok(QubitOp::Swap(a_indices, b_indices))
    }
}

/// Make a Control QubitOp
///
/// # Example
/// ```
/// use qip::state_ops::make_control_op;
/// use qip::state_ops::QubitOp::{Matrix, Control};
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
pub fn make_control_op(mut c_indices: Vec<u64>, op: QubitOp) -> Result<QubitOp, &'static str> {
    if c_indices.is_empty() {
        Err("Must supply at least one control index")
    } else {
        match op {
            QubitOp::Control(oc_indices, oo_indices, op) => {
                c_indices.extend(oc_indices);
                Ok(QubitOp::Control(c_indices, oo_indices, op))
            }
            op => {
                let o_indices = (0..num_indices(&op)).map(|i| get_index(&op, i)).collect();
                Ok(QubitOp::Control(c_indices, o_indices, Box::new(op)))
            }
        }
    }
}

/// Make a Function QubitOp
pub fn make_function_op(
    input_indices: Vec<u64>,
    output_indices: Vec<u64>,
    f: Box<Fn(u64) -> (u64, f64) + Send + Sync>,
) -> Result<QubitOp, &'static str> {
    if input_indices.is_empty() || output_indices.is_empty() {
        Err("Input and Output indices must not be empty")
    } else {
        Ok(QubitOp::Function(input_indices, output_indices, f))
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
    (0..nindices).fold(0, |acc, j| -> u64 {
        let indx = mat_indices[j as usize];
        let bit = get_bit(full_index, n - 1 - indx);
        set_bit(acc, nindices - 1 - j, bit)
    })
}

/// Given the `sub_index` for the submatrix, and a base to overwrite values, get the full index for the 2^n by 2^n matrix.
pub fn sub_to_full(n: u64, mat_indices: &[u64], sub_index: u64, base: u64) -> u64 {
    let nindices = mat_indices.len() as u64;
    (0..nindices).fold(base, |acc, j| {
        let indx = mat_indices[j as usize];
        let bit = get_bit(sub_index, nindices - 1 - j);
        set_bit(acc, n - 1 - indx, bit)
    })
}

/// Get the number of indices represented by `op`
pub fn num_indices(op: &QubitOp) -> usize {
    match &op {
        QubitOp::Matrix(indices, _) => indices.len(),
        QubitOp::Swap(a, b) => a.len() + b.len(),
        QubitOp::Control(cs, os, _) => cs.len() + os.len(),
        QubitOp::Function(inputs, outputs, _) => inputs.len() + outputs.len(),
    }
}

/// Get the `i`th qubit index for `op`
pub fn get_index(op: &QubitOp, i: usize) -> u64 {
    match &op {
        QubitOp::Matrix(indices, _) => indices[i],
        QubitOp::Swap(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        QubitOp::Control(cs, os, _) => {
            if i < cs.len() {
                cs[i]
            } else {
                os[i - cs.len()]
            }
        }
        QubitOp::Function(inputs, outputs, _) => {
            if i < inputs.len() {
                inputs[i]
            } else {
                outputs[i - inputs.len()]
            }
        }
    }
}


/// Convert &QubitOp to equivalent PrecisionQubitOp<P>
fn clone_as_precision_op<P: Precision>(op: &QubitOp) -> PrecisionQubitOp<P> {
    match op {
        QubitOp::Matrix(indices, data) => {
            let data: Vec<Complex<P>> = data
                .iter()
                .map(|c| Complex {
                    re: P::from(c.re).unwrap(),
                    im: P::from(c.im).unwrap(),
                })
                .collect();
            Matrix(indices.clone(), data)
        }
        QubitOp::Swap(a_indices, b_indices) => Swap(a_indices.clone(), b_indices.clone()),
        QubitOp::Control(c_indices, o_indices, op) => Control(
            c_indices.clone(),
            o_indices.clone(),
            Box::new(clone_as_precision_op(op)),
        ),
        QubitOp::Function(inputs, outputs, f) => Function(inputs.clone(), outputs.clone(), f),
    }
}

/// Apply `op` to the `input`, storing the results in `output`. If either start at a nonzero state
/// index in their 0th index, use `input/output_offset`.
pub fn apply_op<P: Precision>(
    n: u64,
    op: &QubitOp,
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
                Complex::default()
            } else {
                let vecrow = colbits - input_offset;
                if vecrow >= input.len() as u64 {
                    Complex::default()
                } else {
                    val * input[vecrow as usize]
                }
            }
        };

        // Get value for row and assign
        *outputloc = sum_for_op_cols(nindices, matrow, &[&op], f);
    };

    // Generate output for each output row
    if multithread {
        output.par_iter_mut().enumerate().for_each(row_fn);
    } else {
        output.iter_mut().enumerate().for_each(row_fn);
    }
}

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix<P: Precision>(
    n: u64,
    op: &QubitOp,
    multithread: bool,
) -> Vec<Vec<Complex<P>>> {
    let zeros: Vec<P> = (0..1 << n).map(|_| P::zero()).collect();
    (0..1 << n)
        .map(|i| {
            let mut input = from_reals(&zeros);
            let mut output = input.clone();
            input[i] = Complex {
                re: P::one(),
                im: P::zero(),
            };
            apply_op(n, op, &input, &mut output, 0, 0, multithread);
            output.clone()
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
        let op = QubitOp::Matrix(vec![0, 1, 2], vec![]);
        assert_eq!(num_indices(&op), 3);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
    }

    #[test]
    fn test_get_index_condition() {
        let mop = QubitOp::Matrix(vec![2, 3], vec![]);
        let op = make_control_op(vec![0, 1], mop).unwrap();
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_get_index_swap() {
        let op = QubitOp::Swap(vec![0, 1], vec![2, 3]);
        assert_eq!(num_indices(&op), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_apply_identity() {
        let op = QubitOp::Matrix(vec![0], from_reals(&[1.0, 0.0, 0.0, 1.0]));
        let input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat() {
        let op = QubitOp::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0, false);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat_first() {
        let op = QubitOp::Matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));

        let input = from_reals(&[1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = QubitOp::Matrix(vec![1], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0, false);

        let expected = from_reals(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(expected, output);
    }
}
