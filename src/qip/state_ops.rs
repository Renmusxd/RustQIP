extern crate num;
extern crate rayon;

use num::complex::Complex;
use rayon::prelude::*;

#[derive(Debug)]
pub enum QubitOp {
    MatrixOp(Vec<u64>, Vec<Complex<f64>>),
    SwapOp(Vec<u64>, Vec<u64>),
    ControlOp(Vec<u64>, Box<QubitOp>),
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

/// Set the `bit_index` bit in `num` to `value`.
///
/// # Example
/// ```
/// use qip::state_ops::set_bit;
/// let n = set_bit(0, 1, true);
/// assert_eq!(n, 2);
/// ```
pub fn set_bit(num: u64, bit_index: u64, value: bool) -> u64 {
    let v = 1 << bit_index;
    if value {
        num | v
    } else {
        num & !v
    }
}

/// Get the `bit_index` bit value from `num`.
///
/// # Example
/// ```
/// use qip::state_ops::get_bit;
/// let n = get_bit(2, 1);
/// assert_eq!(n, true);
/// ```
pub fn get_bit(num: u64, bit_index: u64) -> bool {
    ((num >> bit_index) & 1) != 0
}

fn get_flat_index(nindices: u64, i: u64, j: u64) -> u64 {
    let mat_side = 1 << nindices;
    (i * mat_side) + j
}

/// If `op` were a matrix, get the value of M(i,j)
fn get_mat_entry(nindices: u64, i: u64, j: u64, op: &QubitOp) -> Complex<f64> {
    match &op {
        QubitOp::MatrixOp(_, dat) => {
            let flat_index = get_flat_index(nindices, i, j) as usize;
            dat[flat_index]
        }
        QubitOp::SwapOp(_, _) => {
            let n = nindices >> 1;
            let lower_mask: u64 = !(std::u64::MAX << n);

            let row_select = i & lower_mask;
            let row_criteria = row_select == (j >> n);

            let col_select = j & lower_mask;
            let col_criteria = col_select == (i >> n);

            if row_criteria && col_criteria {
                Complex::<f64> {
                    re: 1.0,
                    im: 0.0,
                }
            } else {
                Complex::<f64> {
                    re: 0.0,
                    im: 0.0,
                }
            }
        }
        QubitOp::ControlOp(cqs, op) => {
            let num_op_indices = nindices - cqs.len() as u64;
            let index_threshold = (1 << nindices) - (1 << num_op_indices);
            if i >= index_threshold && j >= index_threshold {
                get_mat_entry(nindices - cqs.len() as u64,
                              i - index_threshold,
                              j - index_threshold,
                              op)
            } else {
                if i == j {
                    Complex::<f64> {
                        re: 1.0,
                        im: 0.0,
                    }
                } else {
                    Complex::<f64> {
                        re: 0.0,
                        im: 0.0,
                    }
                }
            }
        }
    }
}

/// Get the number of indices represented by `op`
fn num_indices(op: &QubitOp) -> usize {
    match &op {
        QubitOp::MatrixOp(indices, _) => indices.len(),
        QubitOp::SwapOp(a, b) => a.len() + b.len(),
        QubitOp::ControlOp(cs, op) => cs.len() + num_indices(op)
    }
}

/// Get the `i`th qubit index for `op`
fn get_index(op: &QubitOp, i: usize) -> u64 {
    match &op {
        QubitOp::MatrixOp(indices, _) => indices[i],
        QubitOp::SwapOp(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        QubitOp::ControlOp(cs, op) => {
            if i < cs.len() {
                cs[i]
            } else {
                get_index(op, i - cs.len())
            }
        }
    }
}

/// Multiply the matrix entries
/// This acts the same as kron product between all the matrices with ones(2^m, 2^m) occupying
/// the space of all missing indices m = n - len(all_indices).
/// This function allows the op indices to be passed in manually to reduce recomputation.
fn multiply_matrix_entries_with_indices(n: u64, row: u64, col: u64, matrices: &Vec<(&Vec<u64>, &QubitOp)>) -> Complex<f64> {
    let mut p = Complex::<f64> {
        re: 1.0,
        im: 0.0,
    };
    for i in 0..matrices.len() {
        let (indices, mat) = &matrices[i];
        let nindices = indices.len() as u64;

        let (x, y) = (0..nindices).fold((0, 0), |acc, j| -> (u64, u64) {
            let (x, y) = acc;
            let indx = indices[j as usize];
            let rowbit = get_bit(row, n - 1 - indx);
            let colbit = get_bit(col, n - 1 - indx);
            let x = set_bit(x, nindices - 1 - j, rowbit);
            let y = set_bit(y, nindices - 1 - j, colbit);
            (x, y)
        });

        let mat_entry = get_mat_entry(nindices, x, y, mat);
        if mat_entry.re == 0.0 && mat_entry.im == 0.0 {
            return Complex::<f64> {
                re: 0.0,
                im: 0.0,
            };
        }
        p = p * mat_entry;
    }
    p
}

/// Calculate the indices for each op in `matrices` and call `multiply_matrix_entries_with_indices`
fn multiply_matrix_entries(n: u64, row: u64, col: u64, matrices: &Vec<&QubitOp>) -> Complex<f64> {
    let mat_indices: Vec<Vec<u64>> = matrices.iter().map(|op| -> Vec<u64> {
        (0..num_indices(op)).map(|i| get_index(op, i)).collect()
    }).collect();
    let mats_and_indices = mat_indices.iter().zip(matrices.iter().cloned()).collect();
    multiply_matrix_entries_with_indices(n, row, col, &mats_and_indices)
}

// TODO doc
pub fn apply_matrices(n: u64, matrices: &Vec<&QubitOp>,
                      input: &Vec<Complex<f64>>, output: &mut Vec<Complex<f64>>,
                      input_offset: u64, output_offset: u64) {
    let mut flat_indices: Vec<u64> = matrices.iter().map(|item| -> Vec<u64> {
        (0..num_indices(item)).map(|i| -> u64 {
            let indx = get_index(item, i);
            assert!(indx < n);
            indx
        }).collect()
    }).flatten().collect();
    flat_indices.sort();
    let flat_indices = flat_indices;
    let nindices: u64 = flat_indices.len() as u64;

    let mat_indices: Vec<Vec<u64>> = matrices.iter().map(|op| -> Vec<u64> {
        (0..num_indices(op)).map(|i| get_index(op, i)).collect()
    }).collect();
    let mats_and_indices = mat_indices.iter().zip(matrices.iter().cloned()).collect();

    // Generate output for each output row
    output.par_iter_mut().enumerate().for_each(|entry| {
        let (outputrow, outputloc) = entry;
        let row = output_offset + (outputrow as u64);

        // Get value for row and assign
        *outputloc = (0..1 << nindices).map(|i| -> Complex<f64> {
            let colbits = (0..nindices).fold(row as u64, |acc, j| {
                let indx = flat_indices[j as usize];
                let bit_val = get_bit(i, nindices - 1 - j);
                set_bit(acc, n - 1 - indx, bit_val)
            });
            if colbits < input_offset {
                return Complex::<f64> {
                    re: 0.0,
                    im: 0.0,
                };
            }
            let vecrow = colbits - input_offset;

            if vecrow >= input.len() as u64 {
                return Complex::<f64> {
                    re: 0.0,
                    im: 0.0,
                };
            }

            let p = multiply_matrix_entries_with_indices(n, row, colbits, &mats_and_indices);
            p * input[vecrow as usize]
        }).sum();
    });
}

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
pub fn make_op_matrix(n: u64, ops: &Vec<&QubitOp>) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n).map(|i| {
        let mut input = from_reals(&zeros);
        let mut output = input.clone();
        input[i] = Complex::<f64> {
            re: 1.0,
            im: 0.0,
        };
        apply_matrices(n, ops, &input, &mut output, 0, 0);
        output.clone()
    }).collect()
}

#[cfg(test)]
mod state_ops_tests {
    use super::*;
    use super::QubitOp::*;

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
    fn test_multiply_matrix_entries_single() {
        let mut mat_vec = vec![];
        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(MatrixOp(vec![0], mat));

        let mat_ref = mat_vec.iter().collect();

        assert_eq!(multiply_matrix_entries(1, 0, 0, &mat_ref), Complex::<f64> {
            re: 1.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 0, 1, &mat_ref), Complex::<f64> {
            re: 2.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 1, 0, &mat_ref), Complex::<f64> {
            re: 3.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 1, 1, &mat_ref), Complex::<f64> {
            re: 4.0,
            im: 0.0,
        });
    }

    #[test]
    fn test_multiply_matrix_entries_multi() {
        let mut mat_vec = vec![];
        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(MatrixOp(vec![0], mat));

        let mat = from_reals(&vec![1.0, 10.0, 100.0, 1000.0]);
        let mults = mat.clone();
        mat_vec.push(MatrixOp(vec![1], mat));

        let mat_ref = mat_vec.iter().collect();

        for i in 0..2 {
            for j in 0..2 {
                let mult = mults[get_flat_index(1, i, j) as usize].re;

                assert_eq!(multiply_matrix_entries(2, 0 + i, 0 + j, &mat_ref), Complex::<f64> {
                    re: 1.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 0 + i, 2 + j, &mat_ref), Complex::<f64> {
                    re: 2.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 0 + j, &mat_ref), Complex::<f64> {
                    re: 3.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 2 + j, &mat_ref), Complex::<f64> {
                    re: 4.0 * mult,
                    im: 0.0,
                });
            }
        }
    }

    #[test]
    fn test_multiply_matrix_entries_multi_rev() {
        let mut mat_vec = vec![];
        let mat = from_reals(&vec![1.0, 10.0, 100.0, 1000.0]);
        let mults = mat.clone();
        mat_vec.push(MatrixOp(vec![1], mat));

        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(MatrixOp(vec![0], mat));

        let mat_ref = mat_vec.iter().collect();

        for i in 0..2 {
            for j in 0..2 {
                let mult = mults[get_flat_index(1, i, j) as usize].re;

                assert_eq!(multiply_matrix_entries(2, 0 + i, 0 + j, &mat_ref), Complex::<f64> {
                    re: 1.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 0 + i, 2 + j, &mat_ref), Complex::<f64> {
                    re: 2.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 0 + j, &mat_ref), Complex::<f64> {
                    re: 3.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 2 + j, &mat_ref), Complex::<f64> {
                    re: 4.0 * mult,
                    im: 0.0,
                });
            }
        }
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
        let op = ControlOp(vec![0, 1], Box::new(mop));
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
        apply_matrices(1, &vec![&op], &input, &mut output, 0, 0);

        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap() {
        let op = MatrixOp(vec![0], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let mut input = from_reals(&vec![1.0, 0.0]);
        let mut output = from_reals(&vec![0.0, 0.0]);
        apply_matrices(1, &vec![&op], &input, &mut output, 0, 0);

        input.reverse();
        assert_eq!(input, output);
    }

    #[test]
    fn test_apply_swap_first() {
        let op = MatrixOp(vec![0], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let input = from_reals(&vec![1.0, 0.0, 0.0, 0.0]);
        let mut output = from_reals(&vec![0.0, 0.0, 0.0, 0.0]);
        apply_matrices(2, &vec![&op], &input, &mut output, 0, 0);

        let expected = from_reals(&vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(expected, output);

        let op = MatrixOp(vec![1], from_reals(&vec![0.0, 1.0, 1.0, 0.0]));
        let mut output = from_reals(&vec![0.0, 0.0, 0.0, 0.0]);
        apply_matrices(2, &vec![&op], &input, &mut output, 0, 0);

        let expected = from_reals(&vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(expected, output);
    }
}