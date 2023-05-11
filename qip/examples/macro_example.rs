#[cfg(feature = "macros")]
use qip::prelude::*;
#[cfg(feature = "macros")]
use qip_macros::program;

#[cfg(feature = "macros")]
fn gamma<P: Precision, CB: CliffordTBuilder<P>>(
    _cb: &mut CB,
    ra: CB::Register,
    rb: CB::Register,
) -> Result<(CB::Register, CB::Register), CircuitError> {
    Ok((ra, rb))
}

#[cfg(feature = "macros")]
fn main() -> Result<(), CircuitError> {
    let mut b = LocalBuilder::<f64>::default();
    let ra = b.qudit(3).unwrap();
    let rb = b.qudit(3).unwrap();

    let (_ra, _rb) = program!(&mut b; ra, rb;
        // Applies gamma to |ra[0] ra[1]>|ra[2]>
        gamma ra[0..2], ra[2];
        // Applies gamma to |ra[0] rb[0]>|ra[2]>
        gamma [ra[0], rb[0]], ra[2];
        // Applies gamma to |ra[0]>|rb[0] ra[2]>
        gamma ra[0], [rb[0], ra[2]];
        // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |111>
        control gamma rb, ra[0..2], ra[2];
        // Applies gamma to |ra[0] ra[1]>|ra[2]> if rb == |110> (rb[0] == |0>, rb[1] == 1, ...)
        control(0b110) gamma rb, ra[0..2], ra[2];
    )?;

    Ok(())
}

#[cfg(not(feature = "macros"))]
fn main() {
    panic!("Macros not enabled.")
}
