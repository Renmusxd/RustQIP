use crate::errors::CircuitError;
use crate::unitary_decomposition::utils::gray_code;

pub struct BitPather {
    encoding: Vec<u64>,
    reverse_lookup: Vec<u64>,
}

impl BitPather {
    pub(crate) fn new(n: u64) -> Self {
        let encoding = gray_code(n);
        let mut reverse_lookup = vec![];
        reverse_lookup.resize(encoding.len(), 0);

        encoding.iter().enumerate().for_each(|(indx, code)| {
            reverse_lookup[*code as usize] = indx as u64;
        });

        Self {
            encoding,
            reverse_lookup,
        }
    }

    /// Take the list of indices with nonzero values and return the path through them
    /// to the target, returns the bits needed to swap (in the form `1 << index`).
    pub fn path(&self, to: u64, _through: &[u64]) -> Result<Vec<(u64, u64)>, CircuitError> {
        if to as usize > self.reverse_lookup.len() {
            CircuitError::make_err(format!(
                "Value to={:?} is greater than encoding length {:?}",
                to,
                self.reverse_lookup.len()
            ))
        } else {
            let target_index = self.reverse_lookup[to as usize] as usize;
            Ok(self.encoding[target_index..]
                .iter()
                .zip(self.encoding[target_index + 1..].iter())
                .map(|(a, b)| (*b, *a))
                .rev()
                .collect())
        }
    }
}
