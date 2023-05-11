#[cfg(feature = "parallel")]
pub(crate) use rayon::prelude::*;

use qip_iterators::into_iter;

use std::sync::{Arc, Mutex};

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

/// Extracts bits from a number in a particular order.
///
/// # Example
///
/// ```
/// use qip::utils::extract_bits;
///
/// assert_eq!(extract_bits(0b1010, &[3, 0]), 0b01);
/// ```
#[inline]
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
