/// Get the index into an Op matrix
#[inline]
pub fn get_flat_index(nindices: usize, i: usize, j: usize) -> usize {
    let mat_side = 1 << nindices;
    (i * mat_side) + j
}

/// Flips the bits in `num` from `i`th position to `(n-i)`th position.
///
/// # Example
///
/// ```
/// use qip_iterators::utils::flip_bits;
///
/// assert_eq!(flip_bits(3, 0b100), 0b001);
/// assert_eq!(flip_bits(3, 0b010), 0b010);
/// assert_eq!(flip_bits(4, 0b1010), 0b0101);
/// ```
#[inline]
pub fn flip_bits(n: usize, num: usize) -> usize {
    let leading_zeros = 8 * std::mem::size_of::<usize>() - n;
    num.reverse_bits() >> leading_zeros
}
