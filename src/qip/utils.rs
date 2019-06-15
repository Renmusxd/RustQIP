/// Set the `bit_index` bit in `num` to `value`.
///
/// # Example
/// ```
/// use qip::utils::set_bit;
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
/// use qip::utils::get_bit;
/// let n = get_bit(2, 1);
/// assert_eq!(n, true);
/// ```
pub fn get_bit(num: u64, bit_index: u64) -> bool {
    ((num >> bit_index) & 1) != 0
}

pub fn get_flat_index(nindices: u64, i: u64, j: u64) -> u64 {
    let mat_side = 1 << nindices;
    (i * mat_side) + j
}