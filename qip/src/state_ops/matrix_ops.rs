/// Contains functions, structs, and enums for storing and manipulating the quantum state.
use crate::errors::{CircuitError, CircuitResult};
use crate::types::Representation;
use crate::utils::*;
use crate::{Complex, Precision};
use num_traits::One;
use qip_iterators::iterators::*;
use qip_iterators::matrix_ops::apply_op;
use qip_iterators::utils::{flip_bits, get_bit, get_flat_index, set_bit};

/// Make a Matrix MatrixOp
pub fn make_matrix_op<P>(indices: Vec<usize>, dat: Vec<P>) -> CircuitResult<MatrixOp<P>> {
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
        Ok(MatrixOp::Matrix(indices, dat))
    }
}

/// Make a SparseMatrix MatrixOp from a vector of rows (with `(column, value)`).
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `column`
/// and `row` where `row` is the index of `dat`.
pub fn make_sparse_matrix_op<P>(
    indices: Vec<usize>,
    dat: Vec<Vec<(usize, P)>>,
    order: Representation,
) -> CircuitResult<MatrixOp<P>> {
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

        Ok(MatrixOp::SparseMatrix(indices, dat))
    }
}

/// Make a Swap MatrixOp
pub fn make_swap_op<P>(a_indices: Vec<usize>, b_indices: Vec<usize>) -> CircuitResult<MatrixOp<P>> {
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
        let n = a_indices.len();
        let mut indices = a_indices;
        indices.extend(b_indices);
        Ok(MatrixOp::Swap(n, indices))
    }
}

/// Make a Control MatrixOp
pub fn make_control_op<P>(
    mut c_indices: Vec<usize>,
    op: MatrixOp<P>,
) -> CircuitResult<MatrixOp<P>> {
    if c_indices.is_empty() {
        Err(CircuitError::new("Must supply at least one control index"))
    } else {
        let num_c_indices = c_indices.len();
        match op {
            MatrixOp::Control(num_oc_indices, oo_indices, op) => {
                c_indices.extend(oo_indices);
                Ok(MatrixOp::Control(num_c_indices + num_oc_indices, c_indices, op))
            }
            op => {
                c_indices.extend(op.indices());
                Ok(MatrixOp::Control(num_c_indices, c_indices, Box::new(op)))
            }
        }
    }
}

/// Make a vector of vectors of rows (with `(column, value)`) built from a function
/// `f` which takes row numbers.
/// natural_order indicates that the lowest indexed qubit is the least significant bit in `row` and
/// the output `column` from `f`
pub fn make_sparse_matrix_from_function<P, F: Fn(usize) -> Vec<(usize, P)>>(
    n: usize,
    f: F,
    order: Representation,
) -> Vec<Vec<(usize, P)>> {
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

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn invert_op<P: Precision>(op: MatrixOp<Complex<P>>) -> MatrixOp<Complex<P>> {
    conj_op(transpose_op(op))
}

/// Get conjugate of op.
pub fn conj_op<P: Precision>(op: MatrixOp<Complex<P>>) -> MatrixOp<Complex<P>> {
    match op {
        MatrixOp::Matrix(indices, mat) => {
            let mat = mat.into_iter().map(|v| v.conj()).collect();
            MatrixOp::Matrix(indices, mat)
        }
        MatrixOp::SparseMatrix(indices, mat) => {
            let mat = mat
                .into_iter()
                .map(|v| {
                    v.into_iter()
                        .map(|(indx, val)| (indx, val.conj()))
                        .collect()
                })
                .collect();
            MatrixOp::SparseMatrix(indices, mat)
        }
        MatrixOp::Swap(a_indices, b_indices) => MatrixOp::Swap(a_indices, b_indices),
        MatrixOp::Control(c_indices, op_indices, op) => {
            MatrixOp::Control(c_indices, op_indices, Box::new(conj_op(*op)))
        }
    }
}

/// Invert a unitary op (equivalent to conjugate transpose).
pub fn transpose_op<P: Precision>(op: MatrixOp<Complex<P>>) -> MatrixOp<Complex<P>> {
    match op {
        MatrixOp::Matrix(indices, mut mat) => {
            let n = indices.len();
            (0..1 << n).for_each(|row| {
                (0..row).for_each(|col| {
                    mat.swap(get_flat_index(n, row, col), get_flat_index(n, col, row));
                })
            });
            MatrixOp::Matrix(indices, mat)
        }
        MatrixOp::SparseMatrix(indices, mat) => {
            MatrixOp::SparseMatrix(indices, transpose_sparse(mat))
        }
        MatrixOp::Swap(a_indices, b_indices) => MatrixOp::Swap(a_indices, b_indices),
        MatrixOp::Control(c_indices, op_indices, op) => {
            MatrixOp::Control(c_indices, op_indices, Box::new(transpose_op(*op)))
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

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix<P: Precision>(n: usize, op: &MatrixOp<Complex<P>>) -> Vec<Vec<Complex<P>>> {
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
    use qip_iterators::matrix_ops::get_index;
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
        let op = MatrixOp::<Complex<f64>>::new_matrix(vec![0, 1, 2], vec![]);
        assert_eq!(op.num_indices(), 3);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
    }

    #[test]
    fn test_get_index_condition() {
        let mop = MatrixOp::<Complex<f64>>::new_matrix(vec![2, 3], vec![]);
        let op = make_control_op(vec![0, 1], mop).unwrap();
        assert_eq!(op.num_indices(), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_get_index_swap() {
        let op = MatrixOp::<Complex<f64>>::new_swap(vec![0, 1], vec![2, 3]);
        assert_eq!(op.num_indices(), 4);
        assert_eq!(get_index(&op, 0), 0);
        assert_eq!(get_index(&op, 1), 1);
        assert_eq!(get_index(&op, 2), 2);
        assert_eq!(get_index(&op, 3), 3);
    }

    #[test]
    fn test_apply_identity() {
        let op = MatrixOp::<Complex<f64>>::new_matrix(vec![0], from_reals(&[1.0, 0.0, 0.0, 1.0]));
        let input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat() {
        let op = MatrixOp::<Complex<f64>>::new_matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&[1.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0]);
        apply_op(1, &op, &input, &mut output, 0, 0);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_mat_first() {
        let op = MatrixOp::<Complex<f64>>::new_matrix(vec![0], from_reals(&[0.0, 1.0, 1.0, 0.0]));

        let input = from_reals(&[1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&[0.0, 0.0, 0.0, 0.0]);
        apply_op(2, &op, &input, &mut output, 0, 0);

        let expected = from_reals(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = MatrixOp::<Complex<f64>>::new_matrix(vec![1], from_reals(&[0.0, 1.0, 1.0, 0.0]));
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
        if let MatrixOp::SparseMatrix(_, data) = op1 {
            assert_eq!(data, expected_dat);
        }
        if let MatrixOp::SparseMatrix(_, data) = op2 {
            assert_eq!(data, expected_dat);
        }
    }
}
