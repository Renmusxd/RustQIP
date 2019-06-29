/// This whole module is sort of unused, in the pipeline we will only apply a single op
/// at a time. This is here to help with some debugging and to double check incremental work.

extern crate num;
extern crate rayon;

use num::complex::Complex;
use rayon::prelude::*;

use crate::state_ops::*;
use crate::utils::*;

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
                Complex {
                    re: 1.0,
                    im: 0.0,
                }
            } else {
                Complex {
                    re: 0.0,
                    im: 0.0,
                }
            }
        }
        QubitOp::ControlOp(cqs, oqs, op) => {
            let index_threshold = (1 << nindices) - (1 << oqs.len() as u64);
            if i >= index_threshold && j >= index_threshold {
                get_mat_entry(nindices - cqs.len() as u64,
                              i - index_threshold,
                              j - index_threshold,
                              op)
            } else {
                if i == j {
                    Complex {
                        re: 1.0,
                        im: 0.0,
                    }
                } else {
                    Complex {
                        re: 0.0,
                        im: 0.0,
                    }
                }
            }
        }
    }
}

/// Multiply the matrix entries
/// This acts the same as kron product between all the matrices with ones(2^m, 2^m) occupying
/// the space of all missing indices m = n - len(all_indices).
/// This function allows the op indices to be passed in manually to reduce recomputation.
fn multiply_matrix_entries_with_indices(n: u64, row: u64, col: u64, matrices: &Vec<(&Vec<u64>, &QubitOp)>) -> Complex<f64> {
    let mut p = Complex {
        re: 1.0,
        im: 0.0,
    };
    for i in 0..matrices.len() {
        let (indices, mat) = &matrices[i];
        let nindices = indices.len() as u64;

        let (x,y) = select_matrix_coords(n, nindices, indices, row, col);
        let mat_entry = get_mat_entry(nindices, x, y, mat);
        if mat_entry.re == 0.0 && mat_entry.im == 0.0 {
            return Complex {
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

fn apply_ops(n: u64, matrices: &Vec<&QubitOp>,
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
    output.par_iter_mut().enumerate().for_each(|(outputrow, outputloc)| {
        let row = output_offset + (outputrow as u64);

        // Get value for row and assign
        *outputloc = (0..1 << nindices).map(|i| -> Complex<f64> {
            let colbits = (0..nindices).fold(row as u64, |acc, j| {
                let indx = flat_indices[j as usize];
                let bit_val = get_bit(i, nindices - 1 - j);
                set_bit(acc, n - 1 - indx, bit_val)
            });
            if colbits < input_offset {
                return Complex {
                    re: 0.0,
                    im: 0.0,
                };
            }
            let vecrow = colbits - input_offset;

            if vecrow >= input.len() as u64 {
                return Complex {
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
pub fn make_ops_matrix(n: u64, ops: &Vec<&QubitOp>) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n).map(|i| {
        let mut input = from_reals(&zeros);
        let mut output = input.clone();
        input[i] = Complex {
            re: 1.0,
            im: 0.0,
        };
        apply_ops(n, ops, &input, &mut output, 0, 0);
        output.clone()
    }).collect()
}

#[cfg(tests)]
mod multi_ops_tests {
    use crate::state_ops::{from_reals, QubitOp};

    use super::*;

    #[test]
    fn test_multiply_matrix_entries_single() {
        let mut mat_vec = vec![];
        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(QubitOp::MatrixOp(vec![0], mat));

        let mat_ref = mat_vec.iter().collect();

        assert_eq!(multiply_matrix_entries(1, 0, 0, &mat_ref), Complex {
            re: 1.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 0, 1, &mat_ref), Complex {
            re: 2.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 1, 0, &mat_ref), Complex {
            re: 3.0,
            im: 0.0,
        });

        assert_eq!(multiply_matrix_entries(1, 1, 1, &mat_ref), Complex {
            re: 4.0,
            im: 0.0,
        });
    }

    #[test]
    fn test_multiply_matrix_entries_multi() {
        let mut mat_vec = vec![];
        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(QubitOp::MatrixOp(vec![0], mat));

        let mat = from_reals(&vec![1.0, 10.0, 100.0, 1000.0]);
        let mults = mat.clone();
        mat_vec.push(QubitOp::MatrixOp(vec![1], mat));

        let mat_ref = mat_vec.iter().collect();

        for i in 0..2 {
            for j in 0..2 {
                let mult = mults[get_flat_index(1, i, j) as usize].re;

                assert_eq!(multiply_matrix_entries(2, 0 + i, 0 + j, &mat_ref), Complex {
                    re: 1.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 0 + i, 2 + j, &mat_ref), Complex {
                    re: 2.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 0 + j, &mat_ref), Complex {
                    re: 3.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 2 + j, &mat_ref), Complex {
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
        mat_vec.push(QubitOp::MatrixOp(vec![1], mat));

        let mat = from_reals(&vec![1.0, 2.0, 3.0, 4.0]);
        mat_vec.push(QubitOp::MatrixOp(vec![0], mat));

        let mat_ref = mat_vec.iter().collect();

        for i in 0..2 {
            for j in 0..2 {
                let mult = mults[get_flat_index(1, i, j) as usize].re;

                assert_eq!(multiply_matrix_entries(2, 0 + i, 0 + j, &mat_ref), Complex {
                    re: 1.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 0 + i, 2 + j, &mat_ref), Complex {
                    re: 2.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 0 + j, &mat_ref), Complex {
                    re: 3.0 * mult,
                    im: 0.0,
                });

                assert_eq!(multiply_matrix_entries(2, 2 + i, 2 + j, &mat_ref), Complex {
                    re: 4.0 * mult,
                    im: 0.0,
                });
            }
        }
    }
}