use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{Complex, OpBuilder, Register, UnitaryBuilder};

/// Add some common condition circuits to the UnitaryBuilder structs.
pub trait ConditionBuilder {
    /// A controlled x, using `cr` as control and `r` as input.
    fn cx(&mut self, cr: Register, r: Register) -> (Register, Register);
    /// A controlled y, using `cr` as control and `r` as input.
    fn cy(&mut self, cr: Register, r: Register) -> (Register, Register);
    /// A controlled z, using `cr` as control and `r` as input.
    fn cz(&mut self, cr: Register, r: Register) -> (Register, Register);
    /// A controlled not, using `cr` as control and `r` as input.
    fn cnot(&mut self, cr: Register, r: Register) -> (Register, Register);
    /// Swap `ra` and `rb` controlled by `cr`.
    fn cswap(
        &mut self,
        cr: Register,
        ra: Register,
        rb: Register,
    ) -> Result<(Register, Register, Register), CircuitError>;
    /// Apply a unitary matrix to the register. If mat is 2x2 then can broadcast to all qubits.
    fn cmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<(Register, Register), CircuitError>;
    /// Apply a orthonormal matrix to the register. If mat is 2x2 then can broadcast to all qubits.
    fn crealmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: &[f64],
    ) -> Result<(Register, Register), CircuitError>;
}

impl<B: UnitaryBuilder> ConditionBuilder for B {
    fn cx(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(self, cr, rb, |b, r| b.x(r))
    }
    fn cy(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(self, cr, rb, |b, r| b.y(r))
    }
    fn cz(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(self, cr, rb, |b, r| b.z(r))
    }
    fn cnot(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(self, cr, rb, |b, r| b.not(r))
    }
    fn cswap(
        &mut self,
        cr: Register,
        ra: Register,
        rb: Register,
    ) -> Result<(Register, Register, Register), CircuitError> {
        let (cr, (ra, rb)) = try_condition(self, cr, (ra, rb), |b, (ra, rb)| b.swap(ra, rb))?;
        Ok((cr, ra, rb))
    }
    fn cmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<(Register, Register), CircuitError> {
        let (cr, r) = try_condition(self, cr, r, |b, r| b.mat(name, r, mat))?;
        Ok((cr, r))
    }
    fn crealmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: &[f64],
    ) -> Result<(Register, Register), CircuitError> {
        let (cr, r) = try_condition(self, cr, r, |b, r| b.real_mat(name, r, mat))?;
        Ok((cr, r))
    }
}

// Add an implementation for references to the unitary builder so you don't have to do some type
// shenanigans.
impl ConditionBuilder for &mut dyn UnitaryBuilder {
    fn cx(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(*self, cr, rb, |b, r| b.x(r))
    }
    fn cy(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(*self, cr, rb, |b, r| b.y(r))
    }
    fn cz(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(*self, cr, rb, |b, r| b.z(r))
    }
    fn cnot(&mut self, cr: Register, rb: Register) -> (Register, Register) {
        condition(*self, cr, rb, |b, r| b.not(r))
    }
    fn cswap(
        &mut self,
        cr: Register,
        ra: Register,
        rb: Register,
    ) -> Result<(Register, Register, Register), CircuitError> {
        let (cr, (ra, rb)) = try_condition(*self, cr, (ra, rb), |b, (ra, rb)| b.swap(ra, rb))?;
        Ok((cr, ra, rb))
    }
    fn cmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<(Register, Register), CircuitError> {
        let (cr, r) = try_condition(*self, cr, r, |b, r| b.mat(name, r, mat))?;
        Ok((cr, r))
    }
    fn crealmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: &[f64],
    ) -> Result<(Register, Register), CircuitError> {
        let (cr, r) = try_condition(*self, cr, r, |b, r| b.real_mat(name, r, mat))?;
        Ok((cr, r))
    }
}

/// Condition a circuit defined by `f` using `cr`.
pub fn condition<F, RS, OS>(
    b: &mut dyn UnitaryBuilder,
    cr: Register,
    rs: RS,
    f: F,
) -> (Register, OS)
where
    F: FnOnce(&mut dyn UnitaryBuilder, RS) -> OS,
{
    let mut c = b.with_condition(cr);
    let rs = f(&mut c, rs);
    let r = c.release_register();
    (r, rs)
}

/// Condition a circuit defined by `f` using `cr`, better supports Result types.
pub fn try_condition<F, RS, OS>(
    b: &mut dyn UnitaryBuilder,
    cr: Register,
    rs: RS,
    f: F,
) -> Result<(Register, OS), CircuitError>
where
    F: FnOnce(&mut dyn UnitaryBuilder, RS) -> Result<OS, CircuitError>,
{
    let mut c = b.with_condition(cr);
    let rs = f(&mut c, rs)?;
    let r = c.release_register();
    Ok((r, rs))
}

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

/// A helper macro for applying functions to specific qubits in registers.
///
/// Macro takes a &builder, a comma seperated list of qubits and optionally expressions which can be
/// referenced as `&[u64]` - plus a closure which takes a &builder and tuple of registers and
/// returns a tuple of the same size.
///
/// Currently you cannot reuse registers, so `ra` cannot be referenced more than once, if you would
/// like more qubits from the register just put all the indices together and split them out inside
/// the closure (or call `register_expr!` again inside).
///
/// Macro automatically uses `?` on CircuitErrors, you should wrap in a function if you'd like to
/// catch and handle those manually.
///
/// # Example
/// ```
/// use qip::*;
/// # fn main() -> Result<(), CircuitError> {
///
/// let n = 3;
/// let mut b = OpBuilder::new();
/// let ra = b.register(n)?;
/// let rb = b.register(n)?;
/// // Apply NOT to ra[0..n], and rb[1]
/// let (ra, rb) = register_expr!(&mut b, ra, rb[1]; |b, (ra, rb)| {
///   let ra = b.not(ra);
///   let rb = b.not(rb);
///   Ok((ra, rb))
/// });
/// let r = b.merge(vec![ra, rb]);
///
/// Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! register_expr {
    (@splitter($acc:ident) $builder:expr, $name:ident $indices:expr; $($tail:tt)*) => {
        let $name: (Register, Register) = $builder.split($name, &$indices)?;
        $acc.push($name.1);
        let $name = $name.0;
    };
    (@splitter($acc:ident) $builder:expr, $name:ident $indices:expr, $($tail:tt)*) => {
        let $name: (Register, Register) = $builder.split($name, &$indices)?;
        $acc.push($name.1);
        let $name = $name.0;
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };
    (@splitter($acc:ident) $builder:expr, $name:ident; $($tail:tt)*) => {
        // let $name = $name;
    };
    (@splitter($acc:ident) $builder:expr, $name:ident, $($tail:tt)*) => {
        // let $name = $name;
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };

    (@joiner($remaining:ident) $builder:expr, $name:ident $indices:expr; $($tail:tt)*) => {
        let tmp: Vec<Register> = $builder.split_all($name);
        let $name: Register = $builder.merge_with_indices($remaining.pop().unwrap(), tmp, &$indices)?;
    };
    (@joiner($remaining:ident) $builder:expr, $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@joiner($remaining) $builder, $($tail)*);
        let tmp: Vec<Register> = $builder.split_all($name);
        let $name: Register = $builder.merge_with_indices($remaining.pop().unwrap(), tmp, &$indices)?;
    };
    (@joiner($remaining:ident) $builder:expr, $name:ident; $($tail:tt)*) => {
        // let $name = $name
    };
    (@joiner($remaining:ident) $builder:expr, $name:ident, $($tail:tt)*) => {
        register_expr!(@joiner($remaining) $builder, $($tail)*);
        // let $name = $name
    };

    (@as_expr $e:expr) => { $e };
    (@args ($($body:tt)*) <- $name:ident $indices:expr; $($tail:tt)*) => {
        register_expr!(@as_expr ($($body)* $name))
    };
    (@args ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@args ($($body)* $name,) <- $($tail)*)
    };
    (@args ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        register_expr!(@as_expr ($($body)* $name))
    };
    (@args ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@args ($($body)* $name,) <- $($tail)*)
    };

    (@names_out ($($body:tt)*) <- $name:ident $indices:expr; $($tail:tt)*) => {
        register_expr!(@as_expr ($($body)* $name))
    };
    (@names_out ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@names_out ($($body)* $name,) <- $($tail)*)
    };
    (@names_out ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        register_expr!(@as_expr ($($body)* $name))
    };
    (@names_out ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@names_out ($($body)* $name,) <- $($tail)*)
    };

    (@names_in ($($body:tt)*) <- $name:ident $indices:expr; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@names_in ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@names_in ($($body)* $name,) <- $($tail)*)
    };
    (@names_in ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@names_in ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@names_in ($($body)* $name,) <- $($tail)*)
    };

    // Use a wrapper function called run_f in order to force any contained lambda to take on the
    // correct signature.
    (@func($builder:expr, $args:expr) ($($body:tt)*) <- $name:ident $indices:expr; $func:expr) => {
        {
            fn run_f<F>(b: &mut dyn UnitaryBuilder, rs: ($($body)* Register), f: F) ->  Result<($($body)* Register), CircuitError>
            where F: FnOnce(&mut dyn UnitaryBuilder, ($($body)* Register)) -> Result<($($body)* Register), CircuitError> {
                f(b, rs)
            }
            run_f($builder, $args, $func)
        };
    };
    (@func($builder:expr, $args:expr) ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@func($builder, $args) ($($body)* Register,) <- $($tail)*)
    };
    (@func($builder:expr, $args:expr) ($($body:tt)*) <- $name:ident; $func:expr) => {
        {
            fn run_f<F>(b: &mut dyn UnitaryBuilder, rs: ($($body)* Register), f: F) ->  Result<($($body)* Register), CircuitError>
            where F: FnOnce(&mut dyn UnitaryBuilder, ($($body)* Register)) -> Result<($($body)* Register), CircuitError> {
                f(b, rs)
            }
            run_f($builder, $args, $func)
        };
    };
    (@func($builder:expr, $args:expr) ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@func($builder, $args) ($($body)* Register,) <- $($tail)*)
    };

    ($builder:expr, $($tail:tt)*) => {
        {
            // $name is now a tuple of split thing and original
            let mut remaining_qubits = vec![];
            register_expr!(@splitter(remaining_qubits) $builder, $($tail)*);

            let args = register_expr!(@args () <- $($tail)*);
            let register_expr!(@names_in () <- $($tail)*) = register_expr!(@func($builder, args) () <- $($tail)*)?;

            register_expr!(@joiner(remaining_qubits) $builder, $($tail)*);
            register_expr!(@names_out () <- $($tail)*)
        }
    };
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
    let ra = b.merge(all_rs);
    let rb = b.merge(back_rs);

    (ra, rb)
}

#[cfg(test)]
mod common_circuit_tests {
    use super::*;
    use crate::pipeline::make_circuit_matrix;
    use crate::run_debug;

    #[test]
    fn test_work_on() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let r = b.register(3)?;
        let r_indices = r.indices.clone();
        let r = work_on(&mut b, r, &[0], |b, qs| Ok(qs))?;

        assert_eq!(r_indices, r.indices);
        Ok(())
    }

    #[test]
    fn test_work_on_macro() -> Result<(), CircuitError> {
        let n = 2;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let r = register_expr!(&mut b, r[0]; |b, r| {
            let r = b.not(r);
            Ok((r))
        });

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (t_r, r) = b.split(r, &[0])?;
        let t_r = b.not(t_r);
        let r = b.merge_with_indices(r, vec![t_r], &[0])?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_work_on_macro_2() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (ra, rb) = register_expr!(&mut b, ra[0,2], rb[1]; |b, (ra, rb)| {
            let rb = b.not(rb);
            Ok((ra, rb))
        });
        let r = b.merge(vec![ra, rb]);

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (t_rb, rb) = b.split(rb, &[1])?;
        let t_rb = b.not(t_rb);
        let rb = b.merge_with_indices(rb, vec![t_rb], &[1])?;
        let r = b.merge(vec![ra, rb]);
        let basic_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_work_on_macro_3() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (ra, rb) = register_expr!(&mut b, ra[0,2], rb; |b, (ra, rb)| {
            let rb = b.not(rb);
            Ok((ra, rb))
        });
        let r = b.merge(vec![ra, rb]);

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let rb = b.not(rb);
        let r = b.merge(vec![ra, rb]);
        let basic_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }
}
