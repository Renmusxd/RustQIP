#[cfg(feature = "parallel")]
pub(crate) use rayon::prelude::*;

use crate::into_iter;

use std::sync::{Arc, Mutex};

/// Set the `bit_index` bit in `num` to `value`.
///
/// # Example
/// ```
/// use qip::utils::set_bit;
/// let n = set_bit(0, 1, true);
/// assert_eq!(n, 2);
/// ```
pub fn set_bit(num: usize, bit_index: usize, value: bool) -> usize {
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
/// use qip::utils::get_bit;
/// let n = get_bit(2, 1);
/// assert_eq!(n, true);
/// ```
pub fn get_bit(num: usize, bit_index: usize) -> bool {
    ((num >> bit_index) & 1) != 0
}

/// Mixes two bitstreams, `z_bits` and `o_bits`, takes one bit off the lowest position from
/// each to construct the output, `selector` is used to choose which to use. 0 indices `z_bits` a
///
/// # Example
/// ```
/// use qip::utils::entwine_bits;
///
/// let n = 3;
/// let off_bits = 0b01; // 2 bits from off
/// let on_bits = 0b1;  // 1 bit from on
/// let selector = 0b010; // first take from off, then on, then off
/// assert_eq!(entwine_bits(n, selector, off_bits, on_bits), 0b011);
/// ```
pub fn entwine_bits(
    n: usize,
    mut selector: usize,
    mut off_bits: usize,
    mut on_bits: usize,
) -> usize {
    let mut result = 0;

    for i in 0..n {
        if selector & 1 == 0 {
            let bit = off_bits & 1;
            off_bits >>= 1;
            result |= bit << i;
        } else {
            let bit = on_bits & 1;
            on_bits >>= 1;
            result |= bit << i;
        }
        selector >>= 1;
    }

    result
}

/// Get the index into an Op matrix
pub fn get_flat_index(nindices: usize, i: usize, j: usize) -> usize {
    let mat_side = 1 << nindices;
    (i * mat_side) + j
}

/// Flips the bits in `num` from `i`th position to `(n-i)`th position.
///
/// # Example
///
/// ```
/// use qip::utils::flip_bits;
///
/// assert_eq!(flip_bits(3, 0b100), 0b001);
/// assert_eq!(flip_bits(3, 0b010), 0b010);
/// assert_eq!(flip_bits(4, 0b1010), 0b0101);
/// ```
///
pub fn flip_bits(n: usize, num: usize) -> usize {
    let leading_zeros = 64 - n;
    num.reverse_bits() >> leading_zeros
}

/// Extracts bits from a number in a particular order.
///
/// # Example
///
/// ```
/// use qip::utils::extract_bits;
///
/// assert_eq!(extract_bits(0b1010, &[3, 0]), 0b01);
/// ```
///
pub fn extract_bits(num: usize, indices: &[usize]) -> usize {
    indices.iter().enumerate().fold(0, |acc, (i, index)| {
        let bit = (num >> index) & 1;
        acc | (bit << i)
    })
}

/// Transpose a sparse matrix.
pub fn transpose_sparse<T: Sync + Send>(sparse_mat: Vec<Vec<(usize, T)>>) -> Vec<Vec<(usize, T)>> {
    let sparse_len = sparse_mat.len();
    let flat_mat: Vec<_> = into_iter!(sparse_mat)
        // .into_par_iter()
        .enumerate()
        .map(|(row, v)| {
            let v: Vec<_> = v.into_iter().map(|(col, val)| (col, (row, val))).collect();
            v
        })
        .flatten()
        .collect();
    let mut col_mat = <Vec<Arc<Mutex<Vec<(usize, T)>>>>>::new();
    col_mat.resize_with(sparse_len, || Arc::new(Mutex::new(vec![])));
    into_iter!(flat_mat).for_each(|(col, (row, val)): (usize, (usize, T))| {
        col_mat[col].lock().unwrap().push((row, val))
    });
    let col_mat: Vec<_> = into_iter!(col_mat)
        .map(|v| {
            if let Ok(v) = Arc::try_unwrap(v) {
                v.into_inner().unwrap()
            } else {
                panic!()
            }
        })
        .collect();
    into_iter!(col_mat)
        .map(|mut v: Vec<(usize, T)>| {
            v.sort_by_key(|(row, _)| *row);
            v
        })
        .collect()
}
