use num_traits::One;
use std::ops::Mul;

/// Iterator which provides the indices of nonzero columns for a given row for a collection of ops.
#[derive(Debug)]
pub struct MultiOpIterator<'a, P> {
    iter_ns: &'a [usize],
    iter_outputs: &'a [&'a [(usize, P)]],
    curr_poss: Vec<usize>,
    overflow: bool,
}

impl<'a, P> MultiOpIterator<'a, P> {
    /// Build a new iterator using the number of qubits in each sub iterator, and the outputs of
    /// said iterators on a given row.
    pub fn new(
        iter_ns: &'a [usize],
        iter_outputs: &'a [&'a [(usize, P)]],
    ) -> MultiOpIterator<'a, P> {
        let curr_poss: Vec<usize> = iter_ns.iter().map(|_| 0).collect();
        MultiOpIterator {
            iter_ns,
            iter_outputs,
            curr_poss,
            overflow: false,
        }
    }
}

impl<'a, P> Iterator for MultiOpIterator<'a, P>
where
    P: One + Clone + Mul<P>,
{
    type Item = (usize, P);

    fn next(&mut self) -> Option<Self::Item> {
        if self.overflow {
            self.overflow = false;
            None
        } else {
            let init = (0usize, P::one());
            let ret_val = self
                .curr_poss
                .iter()
                .cloned()
                .zip(self.iter_ns.iter().cloned())
                .zip(self.iter_outputs.iter())
                .fold(init, |(acc_col, acc_val), ((cur_pos, n_pos), outs)| {
                    let (col, val) = outs[cur_pos].clone();
                    let acc_col = (acc_col << n_pos) | col;
                    (acc_col, acc_val * val)
                });

            // Iterate through the current positions and increment when needed.
            let mut broke_early = false;
            let pos_iter = self
                .curr_poss
                .iter_mut()
                .rev()
                .zip(self.iter_outputs.iter().rev());

            for (cur_pos, iter_n) in pos_iter {
                *cur_pos += 1;
                if *cur_pos == iter_n.len() {
                    *cur_pos = 0;
                } else {
                    broke_early = true;
                    break;
                }
            }

            // If all poss overflowed, then next output should be None.
            if !broke_early {
                self.overflow = true;
            }
            Some(ret_val)
        }
    }
}

#[cfg(test)]
mod multi_iter_tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_trivial() {
        let one = Complex::one();
        let entry1 = [(1, one)];
        let entry2 = [(0, one)];
        let r_entry: [&[(usize, Complex<f64>)]; 2] = [&entry1, &entry2];
        let ns = [1, 1];
        let it = MultiOpIterator::new(&ns, &r_entry);
        let v: Vec<_> = it.collect();

        assert_eq!(v, vec![(2, Complex { re: 1.0, im: 0.0 })]);
    }

    #[test]
    fn test_nontrivial() {
        let one = Complex::one();
        let entry1 = [(0, one), (1, one)];
        let entry2 = [(0, one)];
        let r_entry: [&[(usize, Complex<f64>)]; 2] = [&entry1, &entry2];
        let ns = [1, 1];
        let it = MultiOpIterator::new(&ns, &r_entry);
        let v: Vec<_> = it.collect();

        assert_eq!(v, vec![(0, Complex::one()), (2, Complex::one())]);
    }

    #[test]
    fn test_nontrivial_other() {
        let one = Complex::one();
        let entry1 = [(0, one)];
        let entry2 = [(0, one), (1, one)];
        let r_entry: [&[(usize, Complex<f64>)]; 2] = [&entry1, &entry2];
        let ns = [1, 1];
        let it = MultiOpIterator::new(&ns, &r_entry);
        let v: Vec<_> = it.collect();

        assert_eq!(v, vec![(0, Complex::one()), (1, Complex::one())]);
    }

    #[test]
    fn test_mat_iterator() {
        let n = 1usize;
        let one = Complex::one();
        let mat: Vec<Vec<f64>> = (0..1 << n)
            .map(|i| -> Vec<f64> {
                let entry = [(1 - i, one)];
                let r_entry: [&[(usize, Complex<f64>)]; 1] = [&entry];
                let ns = [n];
                let it = MultiOpIterator::new(&ns, &r_entry);
                let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
                it.fold(v, |mut v, (indx, _)| {
                    v[indx] = 1.0;
                    v
                })
            })
            .collect();

        let expected = vec![vec![0.0, 1.0], vec![1.0, 0.0]];

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_double_mat_identity() {
        let n = 2usize;
        let one = Complex::one();
        let mat: Vec<Vec<f64>> = (0..1 << n)
            .map(|i| -> Vec<f64> {
                let entry1 = [((i & 2) >> 1, one)];
                let entry2 = [(i & 1, one)];
                let r_entry: [&[(usize, Complex<f64>)]; 2] = [&entry1, &entry2];
                let ns = [1, 1];
                let it = MultiOpIterator::new(&ns, &r_entry);
                let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
                it.fold(v, |mut v, (indx, _)| {
                    v[indx] = 1.0;
                    v
                })
            })
            .collect();

        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_double_mat_swap() {
        let n = 2usize;
        let one = Complex::one();
        let mat: Vec<Vec<f64>> = (0..1 << n)
            .map(|i| -> Vec<f64> {
                let entry1 = [((!i & 2) >> 1, one)];
                let entry2 = [(!i & 1, one)];
                let r_entry: [&[(usize, Complex<f64>)]; 2] = [&entry1, &entry2];
                let ns = [1, 1];
                let it = MultiOpIterator::new(&ns, &r_entry);
                let v: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
                it.fold(v, |mut v, (indx, _)| {
                    v[indx] = 1.0;
                    v
                })
            })
            .collect();

        let expected = vec![
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(mat, expected);
    }
}
