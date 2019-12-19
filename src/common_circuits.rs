use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{OpBuilder, Register, UnitaryBuilder};

/// Extract a set of indices, provide them to a function, then reinsert them in the correct order.
pub fn work_on<F>(
    b: &mut dyn UnitaryBuilder,
    r: Register,
    indices: &[u64],
    f: F,
) -> Result<Register, CircuitError>
where
    F: Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>,
{
    let (selected, remaining) = b.split(r, indices)?;
    let qs = b.split_all(selected);
    let qs = f(b, qs)?;
    if qs.len() != indices.len() {
        CircuitError::make_err(format!(
            "Output number of qubits from function ({}) did not match number of indices ({}).",
            qs.len(),
            indices.len()
        ))
    } else if let Some(remaining) = remaining {
        b.merge_with_indices(remaining, qs, indices)
    } else {
        let mut qs: Vec<(Register, u64)> = qs.into_iter().zip(indices.iter().cloned()).collect();
        qs.sort_by_key(|(_, indx)| *indx);
        let qs = qs.into_iter().map(|(r, _)| r).collect();
        b.merge(qs)
    }
}

/// Makes a pair of Register in the state `|0n>x|0n> + |1n>x|1n>`
/// # Example
/// ```
/// use qip::*;
///
/// let mut b = OpBuilder::new();
/// // Make a total of 6 qubits in state |000000> + |111111> with first 3 qubits given to alice,
/// // and second 3 given to bob: |000>|000> + |111>|111>
/// let (q_alice, q_bob) = qip::epr_pair(&mut b, 3);
/// ```
pub fn epr_pair(b: &mut OpBuilder, n: u64) -> (Register, Register) {
    let m = 2 * n;

    let r = b.qubit();
    let rs = b.register(m - 1).unwrap();

    let r = b.hadamard(r);

    let (r, rs) = b.cnot(r, rs);

    let mut all_rs = vec![r];
    all_rs.extend(b.split_all(rs));

    let back_rs = all_rs.split_off(n as usize);
    let ra = b.merge(all_rs).unwrap();
    let rb = b.merge(back_rs).unwrap();

    (ra, rb)
}

#[cfg(test)]
mod common_circuit_tests {
    use super::*;
    use crate::run_debug;

    #[test]
    fn test_work_on() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let r = b.register(3)?;
        let r_indices = r.indices.clone();
        let r = work_on(&mut b, r, &[0], |_b, qs| Ok(qs))?;
        run_debug(&r)?;

        assert_eq!(r_indices, r.indices);
        Ok(())
    }
}
