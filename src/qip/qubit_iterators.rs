extern crate num;

use num::complex::Complex;

use super::utils::*;

/// Iterator which provides the indices of nonzero columns for a given row of a matrix
pub struct MatrixOpIterator<'a> {
    n: u64,
    data: &'a [Complex<f64>],
    last_col: Option<u64>,
}

impl<'a> MatrixOpIterator<'a> {
    pub fn new(row: u64, n: u64, data: &'a Vec<Complex<f64>>) -> MatrixOpIterator {
        let lower = get_flat_index(n, row, 0) as usize;
        let upper = get_flat_index(n, row, 1 << n) as usize;
        MatrixOpIterator {
            n,
            data: &data[lower .. upper],
            last_col: None,
        }
    }
}

impl<'a> std::iter::Iterator for MatrixOpIterator<'a> {
    type Item = (u64, Complex<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        let pos = if let Some(last_col) = self.last_col {
            last_col + 1
        } else {
            0
        };
        self.last_col = None;
        let zero = Complex { re: 0.0, im: 0.0 };
        for col in pos..(1 << self.n) {
            let val = self.data[col as usize];
            if val != zero {
                self.last_col = Some(col);
                return Some((col, val));
            }
        }
        None
    }
}

/// Iterator which provides the indices of nonzero columns for a given row of a COp
pub struct ControlledOpIterator<It: std::iter::Iterator<Item=(u64, Complex<f64>)>> {
    row: u64,
    index_threshold: u64,
    op_iter: Option<It>,
    last_col: Option<u64>,
}

impl<It: std::iter::Iterator<Item=(u64, Complex<f64>)>> ControlledOpIterator<It> {
    pub fn new<F: FnOnce(u64) -> It>(row: u64, n_control_indices: u64, n_op_indices: u64, iter_builder: F) -> ControlledOpIterator<It> {
        let n_indices = n_control_indices + n_op_indices;
        let index_threshold = (1 << n_indices) - (1 << n_op_indices);
        let op_iter = if row >= index_threshold {
            Some(iter_builder(row - index_threshold))
        } else {
            None
        };
        ControlledOpIterator {
            row,
            index_threshold,
            op_iter,
            last_col: None,
        }
    }
}

impl<It: std::iter::Iterator<Item=(u64, Complex<f64>)>> std::iter::Iterator for ControlledOpIterator<It> {
    type Item = (u64, Complex<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(it) = &mut self.op_iter {
            let cval = it.next();
            self.last_col = cval.map(|(i, _) | i + self.index_threshold);
            cval.map(|(i, val)| (i + self.index_threshold, val))
        } else {
            if let Some(last_col) = self.last_col {
                if last_col < self.row {
                    self.last_col = Some(self.row);
                } else {
                    self.last_col = None;
                }
            } else {
                self.last_col = Some(self.row);
            }
            self.last_col.map(|c| (c, Complex {
                re: 1.0,
                im: 0.0
            }))
        }
    }
}

/// Iterator which provides the indices of nonzero columns for a given row of a SwapOp
pub struct SwapOpIterator {
    row: u64,
    half_n: u64,
    last_col: Option<u64>,
}

impl SwapOpIterator {
    pub fn new(row: u64, n_qubits: u64) -> SwapOpIterator {
        SwapOpIterator {
            row,
            half_n: n_qubits >> 1,
            last_col: None,
        }
    }
}

impl std::iter::Iterator for SwapOpIterator {
    type Item = (u64, Complex<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if let None = self.last_col {
            let lower_mask: u64 = !(std::u64::MAX << self.half_n);
            let lower = self.row & lower_mask;
            let upper = self.row >> self.half_n;
            self.last_col = Some((lower << self.half_n) + upper);
        } else {
            self.last_col = None;
        }
        self.last_col.map(|col| (col, Complex {
            re: 1.0,
            im: 0.0
        }))
    }
}

#[cfg(test)]
mod iterator_tests {
    use crate::state_ops::from_reals;

    use super::*;

    #[test]
    fn test_mat_iterator() {
        let n = 1u64;
        let mat: Vec<Vec<f64>> = (0..1 << n).map(|i| -> Vec<f64> {
            let d = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
            let it = MatrixOpIterator::new(i, n, &d);
            let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
            it.fold(v, |mut v, (indx, _)| {
                v[indx as usize] = 1.0;
                v
            })
        }).collect();

        let expected = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_swap_iterator() {
        let n = 2u64;
        let mat: Vec<Vec<f64>> = (0..1 << n).map(|i| -> Vec<f64> {
            let it = SwapOpIterator::new(i, n);
            let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
            it.fold(v, |mut v, (indx, _)| {
                v[indx as usize] = 1.0;
                v
            })
        }).collect();

        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ];

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_c_iterator() {
        let n = 2u64;

        let d = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
        let builder = |r: u64| {
            MatrixOpIterator::new(r, n >> 1, &d)
        };

        let mat: Vec<Vec<f64>> = (0..1 << n).map(|i| -> Vec<f64> {
            let it = ControlledOpIterator::new(i, n >> 1, n >> 1, builder);
            let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
            it.fold(v, |mut v, (indx, _)| {
                v[indx as usize] = 1.0;
                v
            })
        }).collect();

        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0]
        ];
        assert_eq!(mat, expected);
    }
}

