extern crate rayon;

use crate::{Complex, Precision};
use rayon::prelude::*;
use std::cmp::max;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};

/// Create an array of indices all one bit off from one another.
pub fn gray_code(n: u64) -> Vec<u64> {
    if n == 0 {
        vec![]
    } else if n == 1 {
        vec![0, 1]
    } else {
        let subgray = gray_code(n - 1);
        let reflected: Vec<_> = subgray.clone().into_iter().rev().collect();
        let lhs = subgray;
        let rhs: Vec<_> = reflected.into_iter().map(|x| x | (1 << (n - 1))).collect();
        lhs.into_iter().chain(rhs.into_iter()).collect()
    }
}

/// Transpose a sparse matrix.
pub fn transpose_sparse<T: Sync + Send>(sparse_mat: Vec<Vec<(u64, T)>>) -> Vec<Vec<(u64, T)>> {
    let sparse_len = sparse_mat.len();
    let flat_mat: Vec<_> = sparse_mat
        .into_par_iter()
        .enumerate()
        .map(|(row, v)| {
            let v: Vec<_> = v
                .into_iter()
                .map(|(col, val)| (col, (row as u64, val)))
                .collect();
            v
        })
        .flatten()
        .collect();
    let mut col_mat = <Vec<Arc<Mutex<Vec<(u64, T)>>>>>::new();
    col_mat.resize_with(sparse_len, || Arc::new(Mutex::new(vec![])));
    flat_mat
        .into_par_iter()
        .for_each(|(col, (row, val)): (u64, (u64, T))| {
            col_mat[col as usize].lock().unwrap().push((row, val))
        });
    let col_mat: Vec<_> = col_mat
        .into_par_iter()
        .map(|v| {
            if let Ok(v) = Arc::try_unwrap(v) {
                v.into_inner().unwrap()
            } else {
                panic!()
            }
        })
        .collect();
    col_mat
        .into_par_iter()
        .map(|mut v: Vec<(u64, T)>| {
            v.sort_by_key(|(row, _)| *row);
            v
        })
        .collect()
}

/// Get the value at a given column in a sorted sparse row.
pub fn sparse_value_at_col<C: Ord, T>(col: C, row: &[(C, T)]) -> Option<&T> {
    if row.is_empty() || row[0].0 > col {
        None
    } else if row[0].0 == col {
        Some(&row[0].1)
    } else {
        let res = row
            .binary_search_by(|(c, _)| c.cmp(&col))
            .map(|indx| &row[indx].1);
        match res {
            Ok(indx) => Some(indx),
            Err(_) => None,
        }
    }
}

/// Get the value in a sparse matrix at given coords.
pub fn sparse_value_at_coords<C: Ord, T>(
    row: usize,
    col: C,
    sparse_mat: &[Vec<(C, T)>],
) -> Option<&T> {
    sparse_value_at_col(col, &sparse_mat[row])
}

/// Apply a phase to the column of a sparse matrix.
pub fn apply_phase_to_column<P: Precision>(
    column: u64,
    phi: P,
    sparse_mat: &mut [Vec<(u64, Complex<P>)>],
) {
    let phi = Complex {
        re: P::zero(),
        im: phi,
    };
    let phase = phi.exp();
    sparse_mat.par_iter_mut().for_each(|v| {
        // This is a common one so hard code a check.
        let indx = if !v.is_empty() && v[0].0 == column {
            Ok(0)
        } else {
            v.binary_search_by_key(&column, |(c, _)| *c)
        };

        if let Ok(indx) = indx {
            let val = &mut v[indx].1;
            *val = *val * phase;
        }
    })
}

/// Apply a phase to a row of a sparse matrix.
pub fn apply_phase_to_row<P: Precision>(phi: P, sparse_row: &mut [(u64, Complex<P>)]) {
    let phi = Complex {
        re: P::zero(),
        im: phi,
    };
    let phase = phi.exp();
    sparse_row.par_iter_mut().for_each(|(_, val)| {
        *val = *val * phase;
    })
}

/// Apply a phase to all entries of a sparse matrix.
pub fn apply_global_phase<P: Precision>(phi: P, sparse_mat: &mut [Vec<(u64, Complex<P>)>]) {
    let phi = Complex {
        re: P::zero(),
        im: phi,
    };
    let phase = phi.exp();
    sparse_mat.par_iter_mut().for_each(|v| {
        v.iter_mut().for_each(|(_, val)| {
            *val = *val * phase;
        })
    })
}

/// Apply a rotation matrix to the two given rows. Specifically to the single bit difference between
/// them, controlled by the remaining bits. Then remove all entries in which the value doesn't meet
/// the criteria.
pub fn apply_controlled_rotation_and_clean<
    P: Precision,
    T: Clone + Add<Output = T> + Mul<P, Output = T> + Send + Sync,
    F: Fn(&T) -> bool,
>(
    from_row: u64,
    to_row: u64,
    theta: P,
    sparse_mat: &mut [Vec<(u64, T)>],
    criteria: F,
) {
    apply_controlled_rotation(from_row, to_row, theta, sparse_mat);
    sparse_mat[from_row as usize].retain(|(_, v)| criteria(v));
    sparse_mat[to_row as usize].retain(|(_, v)| criteria(v));
}

/// Apply a rotation matrix to the two given rows. Specifically to the single bit difference between
/// them, controlled by the remaining bits.
pub fn apply_controlled_rotation<
    P: Precision,
    T: Clone + Add<Output = T> + Mul<P, Output = T> + Send + Sync,
>(
    from_row: u64,
    to_row: u64,
    theta: P,
    sparse_mat: &mut [Vec<(u64, T)>],
) {
    // R = c|s0><s0| - s|s0><s1| + s|s1><s0| + c|s1><s1|
    // with c = cos(theta) and s = sin(theta)
    // So apply to the existing sparse mat by finding things which output |s0> and |s1>
    let (s, c) = theta.sin_cos();

    let mut from_vec = std::mem::replace(&mut sparse_mat[from_row as usize], vec![]);
    // Get the things which now output to |s1>, edit those that still output to |s0>
    let mut from_branched: Vec<_> = from_vec
        .par_iter_mut()
        .map(|(col, val)| {
            let v = val.clone() * s;
            *val = val.clone() * c;
            (*col, v)
        })
        .collect();

    let mut to_vec = std::mem::replace(&mut sparse_mat[to_row as usize], vec![]);
    // Get the things which now output to |s0>, edit those that still output to |s1>
    let mut to_branched: Vec<_> = to_vec
        .par_iter_mut()
        .map(|(col, val)| {
            let v = val.clone() * (-s);
            *val = val.clone() * c;
            (*col, v)
        })
        .collect();

    // Sum things which output to |s0>
    merge_vecs(
        &mut from_vec,
        &mut to_branched,
        &mut sparse_mat[from_row as usize],
        |x, y| x + y,
    );
    // Sum things which output to |s1>
    merge_vecs(
        &mut to_vec,
        &mut from_branched,
        &mut sparse_mat[to_row as usize],
        |x, y| x + y,
    );
}

/// Overwrites `vecc` with the merge content of `veca` and `vecb`.
fn merge_vecs<K: Eq + Ord, V, F: Fn(V, V) -> V>(
    veca: &mut Vec<(K, V)>,
    vecb: &mut Vec<(K, V)>,
    out: &mut Vec<(K, V)>,
    acc: F,
) {
    out.clear();
    out.reserve(max(veca.len(), vecb.len()));
    let mut a_item = veca.pop();
    let mut b_item = vecb.pop();
    while (!veca.is_empty()) || (!vecb.is_empty() || a_item.is_some() || b_item.is_some()) {
        let new_item = match (a_item.take(), b_item.take()) {
            (Some((ka, va)), Some((kb, vb))) => {
                if ka > kb {
                    a_item = veca.pop();
                    b_item = Some((kb, vb));
                    (ka, va)
                } else if kb > ka {
                    a_item = Some((ka, va));
                    b_item = vecb.pop();
                    (kb, vb)
                } else {
                    /* if ka == kb */
                    a_item = veca.pop();
                    b_item = vecb.pop();
                    (ka, acc(va, vb))
                }
            }
            (Some((ka, va)), None) => {
                a_item = veca.pop();
                (ka, va)
            }
            (None, Some((kb, vb))) => {
                b_item = vecb.pop();
                (kb, vb)
            }
            (None, None) => unreachable!(),
        };
        out.push(new_item);
    }
    out.reverse();
}

/// Flatten the sparse matrix and add row information.
pub fn flat_sparse<T>(v: Vec<Vec<(u64, T)>>) -> Vec<(u64, u64, T)> {
    v.into_iter()
        .enumerate()
        .map(|(row, v)| -> Vec<(u64, u64, T)> {
            v.into_iter()
                .map(|(col, val)| (row as u64, col, val))
                .collect()
        })
        .flatten()
        .collect()
}

#[cfg(test)]
mod unitary_decomp_tests {
    use super::*;

    fn flat_round(v: Vec<Vec<(u64, f64)>>, prec: i32) -> Vec<(u64, u64, f64)> {
        let flat = flat_sparse(v);
        flat.into_iter()
            .map(|(row, col, val)| {
                let p = 10.0f64.powi(prec);
                (row, col, (val * p).round() / p)
            })
            .filter(|(_, _, v)| *v != 0.0)
            .collect()
    }

    fn test_single_bit_set(mut n: u64) -> bool {
        while n > 0 {
            let lower_bit = (n & 1) == 1;
            n >>= 1;

            if lower_bit {
                if n == 0 {
                    return true;
                }
                break;
            }
        }
        false
    }

    #[test]
    fn test_graycodes() {
        for n in 1..10 {
            let codes = gray_code(n);
            for (i, code) in codes[..codes.len() - 1].iter().enumerate() {
                let next_code = codes[i + 1];
                let diff = *code ^ next_code;
                assert!(test_single_bit_set(diff))
            }
        }
    }

    #[test]
    fn test_transpose() {
        let mat = vec![vec![(0, 1), (1, 2)], vec![(0, 3), (1, 4)]];
        let transpose_mat = transpose_sparse(mat.clone());
        let expected_mat = vec![vec![(0, 1), (1, 3)], vec![(0, 2), (1, 4)]];
        assert_eq!(transpose_mat, expected_mat);
    }

    #[test]
    fn test_merge_noacc() {
        let mut va = vec![(0, 0), (2, 2), (4, 4)];
        let mut vb = vec![(1, 1), (3, 3), (5, 5)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |_, _| panic!());

        let expected: Vec<_> = (0..6).map(|i| (i, i)).collect();
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_acc() {
        let mut va = vec![(0, 0), (1, 1), (2, 2)];
        let mut vb = vec![(0, 0), (1, 1), (2, 2)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_uneven_a_beginning() {
        let mut va = vec![(-1, -1), (0, 0), (1, 1), (2, 2)];
        let mut vb = vec![(0, 0), (1, 1), (2, 2)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let mut expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        expected.insert(0, (-1, -1));
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_uneven_a_end() {
        let mut va = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let mut vb = vec![(0, 0), (1, 1), (2, 2)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let mut expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        expected.push((3, 3));
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_uneven_b_beginning() {
        let mut va = vec![(0, 0), (1, 1), (2, 2)];
        let mut vb = vec![(-1, -1), (0, 0), (1, 1), (2, 2)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let mut expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        expected.insert(0, (-1, -1));
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_uneven_b_end() {
        let mut va = vec![(0, 0), (1, 1), (2, 2)];
        let mut vb = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let mut expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        expected.push((3, 3));
        assert_eq!(v, expected);
    }

    #[test]
    fn test_merge_offsets() {
        let mut va = vec![(-1, -1), (0, 0), (1, 1), (2, 2)];
        let mut vb = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let mut v = vec![];
        merge_vecs(&mut va, &mut vb, &mut v, |x, y| x + y);

        let mut expected: Vec<_> = (0..3).map(|i| (i, 2 * i)).collect();
        expected.insert(0, (-1, -1));
        expected.push((3, 3));
        assert_eq!(v, expected);
    }

    #[test]
    fn test_full_rotation_to_eye() {
        let mut eye = vec![vec![(0, 1.0), (1, 0.0)], vec![(0, 0.0), (1, 1.0)]];
        let expected = eye.clone();
        let theta = std::f64::consts::PI * 2.0;
        apply_controlled_rotation(0, 1, theta, &mut eye);

        let prec = 10;
        let rounded_eye = flat_round(eye, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }

    #[test]
    fn test_half_rotation_to_eye() {
        let mut eye = vec![vec![(0, 1.0), (1, 0.0)], vec![(0, 0.0), (1, 1.0)]];
        let expected = vec![vec![(0, -1.0), (1, 0.0)], vec![(0, 0.0), (1, -1.0)]];
        let theta = std::f64::consts::PI;
        apply_controlled_rotation(0, 1, theta, &mut eye);

        let prec = 10;
        let rounded_eye = flat_round(eye, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }

    #[test]
    fn test_quarter_rotation_to_eye() {
        let mut eye = vec![vec![(0, 1.0), (1, 0.0)], vec![(0, 0.0), (1, 1.0)]];
        let expected = vec![vec![(0, 0.0), (1, -1.0)], vec![(0, 1.0), (1, 0.0)]];
        let theta = std::f64::consts::FRAC_PI_2;
        apply_controlled_rotation(0, 1, theta, &mut eye);

        let prec = 10;
        let rounded_eye = flat_round(eye, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }

    #[test]
    fn test_four_quarter_rotation_to_indices() {
        let mut eye = vec![vec![(0, 1.0), (1, 0.0)], vec![(0, 0.0), (1, 1.0)]];
        let expected = eye.clone();
        let theta = std::f64::consts::FRAC_PI_2;
        apply_controlled_rotation(0, 1, theta, &mut eye);
        apply_controlled_rotation(0, 1, theta, &mut eye);
        apply_controlled_rotation(0, 1, theta, &mut eye);
        apply_controlled_rotation(0, 1, theta, &mut eye);

        let prec = 10;
        let rounded_eye = flat_round(eye, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }

    #[test]
    fn test_rot_quarter() {
        let mut v = vec![
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (1, -std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (1, std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let expected = vec![
            vec![(0, 1.0)],
            vec![(1, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let theta = -std::f64::consts::FRAC_PI_4;
        apply_controlled_rotation(0, 1, theta, &mut v);

        let prec = 10;
        let rounded_eye = flat_round(v, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }

    #[test]
    fn test_rot_quarter_other() {
        let mut v = vec![
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (2, -std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(1, 1.0)],
            vec![
                (0, std::f64::consts::FRAC_1_SQRT_2),
                (2, std::f64::consts::FRAC_1_SQRT_2),
            ],
            vec![(3, 1.0)],
        ];
        let expected = vec![
            vec![(0, 1.0)],
            vec![(1, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let theta = -std::f64::consts::FRAC_PI_4;
        apply_controlled_rotation(0, 2, theta, &mut v);

        let prec = 10;
        let rounded_eye = flat_round(v, prec);
        let rounded_expected = flat_round(expected, prec);
        assert_eq!(rounded_eye, rounded_expected);
    }
}