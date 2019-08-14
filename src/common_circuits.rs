use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{OpBuilder, Qubit, UnitaryBuilder};

/// Condition a circuit defined by `f` using `cq`.
pub fn condition<F, QS>(
    b: &mut dyn UnitaryBuilder,
    cq: Qubit,
    qs: QS,
    f: F,
) -> Result<(Qubit, QS), CircuitError>
where
    F: Fn(&mut dyn UnitaryBuilder, QS) -> Result<QS, CircuitError>,
{
    let mut c = b.with_condition(cq);
    let qs = f(&mut c, qs)?;
    let q = c.release_qubit();
    Ok((q, qs))
}

/// Makes a pair of qubits in the state `|0n>x|0n> + |1n>x|1n>`
pub fn epr_pair(b: &mut OpBuilder, n: u64) -> (Qubit, Qubit) {
    let m = 2 * n;

    let q = b.q(1);
    let qs = b.q(m - 1);

    let q = b.hadamard(q);

    let (q, qs) = condition(b, q, qs, |b, qs| Ok(b.not(qs))).unwrap();

    let mut all_qs = vec![q];
    all_qs.extend(b.split_all(qs));

    let back_qs = all_qs.split_off(n as usize);
    let qa = b.merge(all_qs);
    let qb = b.merge(back_qs);

    (qa, qb)
}
