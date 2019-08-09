use super::decomposition::decompose_unitary;
use crate::{UnitaryBuilder, Complex, Qubit, Precision};

pub fn convert_sparse_to_circuit<P: Precision>(b: &mut dyn UnitaryBuilder, q: Qubit, sparse_unitary: Vec<Vec<(u64, Complex<P>)>>, drop_below: P) -> Result<Qubit, &'static str> {
    let decomposition = decompose_unitary(q.n(), sparse_unitary, drop_below)?;
    let (ops, base) = decomposition.map_err(|| "Decomposition failed.")?;

    let qs = b.split_all(q);

    // TODO apply base controlled.
    // TODO apply all the ops.

    let q = b.merge(qs);
    Ok(q)
}