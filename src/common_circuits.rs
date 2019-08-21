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
    } else {
        b.merge_with_indices(remaining, qs, indices)
    }
}

/// Makes a pair of Register in the state `|0n>x|0n> + |1n>x|1n>`
pub fn epr_pair(b: &mut OpBuilder, n: u64) -> (Register, Register) {
    let m = 2 * n;

    let r = b.r(1);
    let rs = b.r(m - 1);

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
