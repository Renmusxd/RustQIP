use qip::{apply_function, run_local, CircuitError, OpBuilder, Register, UnitaryBuilder};

/// Apply a function f(bool) -> bool as a unitary operator U
///
///         -----
/// |x> --- |   | --- |x>
///         | U |
/// |y> --- |   | --- |y xor f(x)>
///         -----
///
fn apply_oracle<F>(
    b: &mut dyn UnitaryBuilder,
    rx: Register,
    ry: Register,
    f: F,
) -> Result<(Register, Register), CircuitError>
where
    F: 'static + Fn(bool) -> bool + Send + Sync,
{
    let (rx, ry) = apply_function(b, rx, ry, move |input| {
        let fx = f(input == 1);
        (if fx { 1 } else { 0 }, 0.0)
    })?;

    Ok((rx, ry))
}

/// Check if a function f(bool) -> bool is constant
fn is_constant_f<F>(f: F) -> Result<bool, CircuitError>
where
    F: 'static + Fn(bool) -> bool + Send + Sync,
{
    let mut b = OpBuilder::new();

    // prepare |x> = |+>
    let rx = b.qubit();
    let rx = b.hadamard(rx);

    // prepare |x> = |->
    let ry = b.qubit();
    let ry = b.x(ry);
    let ry = b.hadamard(ry);

    let (rx, _ry) = apply_oracle(&mut b, rx, ry, f)?;

    // measure |x> on Hadamard base
    let rx = b.hadamard(rx);
    let (rx, rx_m) = b.measure(rx);

    let (_, measurement) = run_local::<f64>(&rx)?;

    let (result_x, _) = measurement.get_measurement(&rx_m).unwrap();
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
