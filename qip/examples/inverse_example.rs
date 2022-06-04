use qip::inverter::Invertable;
use qip::prelude::*;
use qip_macros::*;

#[invert(gamma_inv, _arg)]
#[cfg(feature = "macros")]
fn gamma<P: Precision, CB: CliffordTBuilder<P> + Invertable<SimilarBuilder=CB>>(
    cb: &mut CB,
    _arg: usize,
    ra: CB::Register,
    rb: CB::Register,
) -> Result<(CB::Register, CB::Register), CircuitError> {
    cb.cnot(ra, rb)
}

#[cfg(feature = "macros")]
fn main() -> Result<(), CircuitError> {
    let mut b = LocalBuilder::<f64>::default();
    let ra = b.qudit(3).unwrap();
    let rb = b.qudit(3).unwrap();

    let (_ra, _rb) = program!(&mut b; ra, rb;
        gamma(10) ra, rb;
        gamma_inv(10) ra, rb;
    )?;

    Ok(())
}

#[cfg(not(feature = "macros"))]
fn main() {}
