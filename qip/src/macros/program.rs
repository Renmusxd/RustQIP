use crate::prelude::CliffordTBuilder;
use crate::Precision;
use std::ops::Range;

/// Negate all the qubits in a register where the mask bit == 0.
pub fn negate_bitmask<P: Precision, CB: CliffordTBuilder<P>>(
    b: &mut CB,
    r: CB::Register,
    mask: u64,
) -> CB::Register {
    let rs = b.split_all_register(r);
    let (rs, _) = rs.into_iter().fold(
        (Vec::default(), mask),
        |(mut qubit_acc, mask_acc), qubit| {
            let lowest = mask_acc & 1;
            let qubit = if lowest == 0 { b.not(qubit) } else { qubit };
            qubit_acc.push(qubit);
            (qubit_acc, mask_acc >> 1)
        },
    );
    b.merge_registers(rs).unwrap()
}

/// Helper for indexing into Qubits
#[derive(Debug)]
pub struct QubitIndices<It>
    where
        It: IntoIterator<Item=usize>,
{
    it: It,
}

impl<It> From<It> for QubitIndices<It>
    where
        It: IntoIterator<Item=usize>,
{
    fn from(it: It) -> Self {
        Self { it }
    }
}

impl<It> IntoIterator for QubitIndices<It>
    where
        It: IntoIterator<Item=usize>,
{
    type Item = It::Item;
    type IntoIter = It::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.it.into_iter()
    }
}

impl From<[Range<usize>; 1]> for QubitIndices<Range<usize>> {
    fn from(it: [Range<usize>; 1]) -> Self {
        let it = it[0].clone();
        Self { it }
    }
}
