use qip::builder::Qudit;
use qip::builder_traits::UnitaryBuilder;
use qip::prelude::*;
use std::num::NonZeroUsize;

// fn apply_oracle<P: Precision>(
//     &mut b: dyn UnitaryBuilder<P>,
//     rx: Qudit,
//     ry: Qudit,
//     f: P,
// ) -> Result<(Qudit, Qudit), CircuitError>

// {
//     // let (rx, ry) = apply_function

// }

fn is_constant_f<P: Precision>(f: P) -> Result<bool, CircuitError> {
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
