use crate::feynman_state::FeynmanPrecisionOp;
use crate::Precision;
use num::{Complex, Zero};

/// A container for memoized feynman state amplitude calculations.
#[derive(Debug)]
pub(crate) struct FeynmanMemory<P: Precision> {
    memory: Vec<Complex<P>>,
}

/// A struct for managing saved values in a `FeynmanState`
impl<P: Precision> FeynmanMemory<P> {
    /// Make a new FeynmanMemory unit, it can optimize positions using `ops`
    pub(crate) fn new(size: usize, _ops: &[FeynmanPrecisionOp<P>]) -> Self {
        Self {
            memory: vec![Complex::zero(); size],
        }
    }

    /// Iterate through amplitudes and their associated indices.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Complex<P>)> {
        self.memory.iter_mut().enumerate()
    }

    /// Get an amplitude if in scope
    pub(crate) fn get(&self, index: usize) -> Option<&Complex<P>> {
        if index < self.memory.len() {
            Some(&self.memory[index])
        } else {
            None
        }
    }
}
