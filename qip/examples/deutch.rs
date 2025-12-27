#[cfg(feature = "macros")]
use qip::builder::Qudit;
#[cfg(feature = "macros")]
use qip::builder_traits::UnitaryBuilder;
#[cfg(feature = "macros")]
use qip::prelude::*;
#[cfg(feature = "macros")]
use std::num::NonZeroUsize;

#[cfg(feature = "macros")]
fn apply_oracle<F>(
    b: &mut dyn UnitaryBuilder,
    rx: Qudit,
    ry: Qudit,
    f: F,
) -> Result<(Qudit, Qudit), CircuitError>
where
    F: 'static + Fn(bool) -> bool + Send + Sync,
{
    let (rx, ry) = program!(b, rx, ry, move |input| {
        let fx = f(input == 1);
        (if fx { 1 } else { 0 }, 0.0)
    }).unwrap();
    Ok((rx, ry))
}

#[cfg(feature = "macros")]
fn is_constant_f<F>(f: F) -> Result<bool, CircuitError>
where
    F: 'static + Fn(bool) -> bool + Send + Sync,
{
    let mut b = LocalBuilder::<f64>::default();

    // prepare |x> = |+>
    let rx = b.qubit();
    let rx = b.h(rx);

    // prepare |x> = |->
    let ry = b.qubit();
    let ry = b.x(ry);
    let ry = b.h(ry);

    let (rx, _ry) = apply_oracle(&mut b, rx, ry, f)?;

    // measure |x> on Hadamard base
    let rx = b.h(rx);
    let (rx, rx_m) = b.measure(rx);

    let (_, measurement) = b.calculate_state();
    let (result_x, _) = measurement.get_measurement(rx_m);

    // function is constant if no phase flip has occurred and |x> == |+>
    Ok(result_x == 0)
}
#[cfg(feature = "macros")]
fn main() -> Result<(), CircuitError> {
    let result = is_constant_f(|x| x)?;
    println!("f(x) = x is constant: {:?}", result);

    let result = is_constant_f(|x| !x)?;
    println!("f(x) = !x is constant: {:?}", result);

    let result = is_constant_f(|_x| true)?;
    println!("f(x) = true is constant: {:?}", result);

    let result = is_constant_f(|_x| false)?;
    println!("f(x) = false is constant: {:?}", result);

    Ok(())
}

#[cfg(not(feature = "macros"))]
fn main() -> () {}
