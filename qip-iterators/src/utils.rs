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

/// Set the `bit_index` bit in `num` to `value`.
///
/// # Example
/// ```
/// use qip_iterators::utils::set_bit;
/// assert_eq!(set_bit(0, 1, true), 2);
/// assert_eq!(set_bit(1, 1, true), 3);
/// assert_eq!(set_bit(1, 0, false), 0);
/// ```
#[inline]
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
/// use qip_iterators::utils::get_bit;
/// let n = get_bit(2, 1);
/// assert_eq!(n, true);
/// ```
#[inline]
pub fn get_bit(num: usize, bit_index: usize) -> bool {
    ((num >> bit_index) & 1) != 0
}
