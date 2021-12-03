use crate::builder_traits::{CliffordTBuilder, RotationsBuilder};
use crate::conditioning::Conditionable;
use crate::errors::CircuitError;
use crate::Precision;

pub fn qfft<P, CB>(b: &mut CB, r: CB::Register) -> Result<CB::Register, CircuitError>
where
    CB: CliffordTBuilder<P> + Conditionable + RotationsBuilder<P>,
    P: Precision,
{
    let mut rs = b
        .split_all_register(r)
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<CB::Register>>>();
    for i in 0..rs.len() {
        let mut ri = rs[i].take().unwrap();
        for j in (i + 1)..rs.len() {
            let rj = rs[j].take().unwrap();
            let mut cb = b.condition_with(rj);
            ri = cb.rz_pi_by(ri, 1 << (j - i))?;
            let rj = cb.dissolve();
            rs[j] = Some(rj);
        }
        let ri = b.h(ri);
        rs[i] = Some(ri);
    }
    for i in 0..rs.len() / 2 {
        let ia = i;
        let ib = rs.len() - 1 - i;
        let ra = rs[ia].take().unwrap();
        let rb = rs[ib].take().unwrap();
        let (ra, rb) = b.swap(ra, rb)?;
        rs[ia] = Some(ra);
        rs[ib] = Some(rb);
    }
    b.merge_registers(rs.into_iter().map(Option::unwrap))
        .ok_or(CircuitError::new("No registers found"))
}
