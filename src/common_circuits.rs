use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{OpBuilder, Register, UnitaryBuilder};

/// Condition a circuit defined by `f` using `cr`.
pub fn condition<F, RS>(
    b: &mut dyn UnitaryBuilder,
    cr: Register,
    rs: RS,
    f: F,
) -> Result<(Register, RS), CircuitError>
where
    F: Fn(&mut dyn UnitaryBuilder, RS) -> Result<RS, CircuitError>,
{
    let mut c = b.with_condition(cr);
    let rs = f(&mut c, rs)?;
    let r = c.release_register();
    Ok((r, rs))
}

/// Makes a pair of Register in the state `|0n>x|0n> + |1n>x|1n>`
pub fn epr_pair(b: &mut OpBuilder, n: u64) -> (Register, Register) {
    let m = 2 * n;

    let r = b.r(1);
    let rs = b.r(m - 1);

    let r = b.hadamard(r);

    let (r, rs) = condition(b, r, rs, |b, rs| Ok(b.not(rs))).unwrap();

    let mut all_rs = vec![r];
    all_rs.extend(b.split_all(rs));

    let back_rs = all_rs.split_off(n as usize);
    let ra = b.merge(all_rs);
    let rb = b.merge(back_rs);

    (ra, rb)
}
