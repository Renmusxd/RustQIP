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
pub fn entwine_bits(n: u64, mut selector: u64, mut off_bits: u64, mut on_bits: u64) -> u64{
    let mut result = 0;

    for i in 0 .. n {
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
pub fn get_flat_index(nindices: u64, i: u64, j: u64) -> u64 {
    let mat_side = 1 << nindices;
    (i * mat_side) + j
}

/// Flips the bits in `num` from ith position to (n-i)th position.
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
pub fn flip_bits(n: usize, num: u64) -> u64 {
    (0 .. n).fold(0, |acc, i| {
        let bit = (num >> i) & 1;
        acc | (bit << (n - 1 - i))
    })
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
pub fn extract_bits(num: u64, indices: &[u64]) -> u64 {
    indices.iter().enumerate().fold(0, |acc, (i, index)| {
        let bit = (num >> index) & 1;
        acc | (bit << i)
    })
}