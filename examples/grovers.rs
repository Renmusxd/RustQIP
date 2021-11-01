use qip::builders::apply_function;
use qip::pipeline::LocalQuantumState;
use qip::*;

/// Take a black box which flips the sign on a specific state |w>.
/// Grovers effectively operates in 2D spaces between the vectors |w> and |w_perp>
/// We initialize in a superposition of all computational basis states: |s>
/// Initially <w|s> = 1/Sqrt(2^n) = sin(theta)
/// and <w_perp|s> = cos(theta)
/// We then repeated apply alternating reflections across |w_perp> and |s>, each pair rotates
/// the state |psi> by 2 theta
/// After T = pi/(4 theta) ~ pi/4 sqrt(2^n) rotations we get <w|psi> ~ sin(pi/2) ~ 1
/// Meaning a measurement of |psi> reveals the marked state |w>
fn prepare_state<P: Precision>(n: u64) -> Result<LocalQuantumState<P>, CircuitError> {
    let mut b = OpBuilder::new();
    let r = b.register(n).unwrap();
    let r = b.hadamard(r);

    let anc = b.qubit();
    let anc = b.not(anc);
    let anc = b.hadamard(anc);

    let r = b.merge(vec![r, anc])?;

    run_local(&r).map(|(s, _)| s)
}

/// Reflection across the |s> vector
fn apply_us(
    b: &mut dyn UnitaryBuilder,
    search: Register,
    ancillary: Register,
) -> Result<(Register, Register), CircuitError> {
    let search = b.hadamard(search);
    let (search, ancillary) = apply_function(b, search, ancillary, |x| {
        (0, if x == 0 { std::f64::consts::PI } else { 0.0 })
    })?;
    let search = b.hadamard(search);
    Ok((search, ancillary))
}

/// Reflection across the |w_perp> vector (oracle).
fn apply_uw(
    b: &mut dyn UnitaryBuilder,
    search: Register,
    ancillary: Register,
    x0: u64,
) -> Result<(Register, Register), CircuitError> {
    // Need to move the x0 value into the closure.
    apply_function(b, search, ancillary, move |x| ((x == x0) as u64, 0.0))
}

/// A reflection across |w_perp> then |s> gives a rotation by 2theta
fn apply_grover_iteration<P: Precision>(
    x: u64,
    s: LocalQuantumState<P>,
) -> Result<LocalQuantumState<P>, CircuitError> {
    let mut b = OpBuilder::new();
    let r = b.register(s.n() - 1)?;
    let anc = b.qubit();

    let (r, anc) = apply_uw(&mut b, r, anc, x)?;
    let (r, _) = apply_us(&mut b, r, anc)?;
    run_with_state(&r, s).map(|(s, _)| s)
}

/// Apply iterations of grovers and plot the chance of successfully finding the marked state.
/// As steps are applied, the system's state rotates in the |w> and |w_perp> plane, when it is
/// aligned and antialigned with |w> the chance of measurement is highest.
/// We will see that the probability of success oscillates with period T = pi/4 sqrt(2^n).
fn main() -> Result<(), CircuitError> {
    let n = 10;
    let x = 42;

    let s = prepare_state::<f64>(n)?;

    let iters = 100;
    let (_, states) = (0..iters).try_fold((s, vec![]), |(s, mut vecs), _| {
        let mut s = apply_grover_iteration(x, s)?;
        let indices: Vec<u64> = (0..n).collect();
        let f = s.stochastic_measure(&indices, 0.0)[x as usize];
        vecs.push(f);
        Ok((s, vecs))
    })?;

    states.into_iter().enumerate().for_each(|(i, f)| {
        println!("{:?}\t{:.*}", i, 5, f);
    });

    Ok(())
}
