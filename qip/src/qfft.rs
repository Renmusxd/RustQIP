use crate::builder_traits::{CliffordTBuilder, RotationsBuilder};
use crate::conditioning::Conditionable;
use crate::errors::{CircuitError, CircuitResult};
use crate::types::Precision;

/// Applies a quantum fourier transform to registers `r`.
pub fn qfft<P, CB>(b: &mut CB, r: CB::Register) -> CircuitResult<CB::Register>
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
        for (j, rj_ref) in rs.iter_mut().enumerate().skip(i + 1) {
            let rj = rj_ref.take().unwrap();
            let mut cb = b.condition_with(rj);
            ri = cb.rz_pi_by(ri, 1 << (j - i))?;
            let rj = cb.dissolve();
            *rj_ref = Some(rj);
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
        .ok_or_else(|| CircuitError::new("No registers found"))
}
