use crate::unitary_decomposition::utils::gray_code;

pub struct BitPather {
    n: u64,
    encoding: Vec<u64>,
}

impl BitPather {
    pub(crate) fn new(n: u64) -> Self {
        Self {
            n,
            encoding: gray_code(n),
        }
    }

    /// Take the list of indices with nonzero values and return the path through them
    /// to the target, returns the bits needed to swap (in the form `1 << index`).
    pub fn path(&self, to: u64, through: &[u64]) -> Result<Vec<(u64, u64)>, &'static str> {
        let target_index = self
            .encoding
            .binary_search(&to)
            .map_err(|_| "Counn't find `to` bits.")?;
        Ok(self.encoding[target_index..]
            .iter()
            .zip(self.encoding[target_index + 1..].iter())
            .map(|(a, b)| (*b, *a))
            .rev()
            .collect())
    }
}
