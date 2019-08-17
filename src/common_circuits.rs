use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{Complex, OpBuilder, Register, UnitaryBuilder};

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
/// })?;
/// let r = b.merge(vec![ra, rb]);
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! register_expr {
    // Split the names into the selected indices, pushing the remaining to the acc vector.
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

    // Join together all the names back to their original indices by pulling from the
    // remaining vector.
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

    // Output (name_1, name_2, ...)
    (@name_tuple ($($body:tt)*) <- $name:ident $indices:expr; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@name_tuple ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@name_tuple ($($body)* $name,) <- $($tail)*)
    };
    (@name_tuple ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@name_tuple ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@name_tuple ($($body)* $name,) <- $($tail)*)
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

    // output (Register, Register, ...) for each name
    (@registers_for_names ($($body:tt)*) <- $name:ident $indices:expr; $($tail:tt)*) => {
        ($($body)* Register)
    };
    (@registers_for_names ($($body:tt)*) <- $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@registers_for_names ($($body)* Register,) <- $($tail)*)
    };
    (@registers_for_names ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        ($($body)* Register)
    };
    (@registers_for_names ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        register_expr!(@registers_for_names ($($body)* Register,) <- $($tail)*)
    };

    // Main public entry point.
    ($builder:expr, $($tail:tt)*) => {
        {
            let run_register = |b: &mut dyn UnitaryBuilder, rs: register_expr!(@registers_for_names () <- $($tail)*)| -> Result<register_expr!(@registers_for_names () <- $($tail)*), CircuitError> {
                let register_expr!(@name_tuple () <- $($tail)*) = rs;
                // $name is now a tuple of split thing and original
                let mut remaining_qubits = vec![];
                register_expr!(@splitter(remaining_qubits) b, $($tail)*);

                let args = register_expr!(@name_tuple () <- $($tail)*);
                let register_expr!(@name_tuple () <- $($tail)*) = register_expr!(@func(b, args) () <- $($tail)*)?;

                register_expr!(@joiner(remaining_qubits) b, $($tail)*);
                Ok(register_expr!(@name_tuple () <- $($tail)*))
            };
            let rs = register_expr!(@name_tuple () <- $($tail)*);
            run_register($builder, rs)
        }
    };
}

macro_rules! program {
    (@splitter($builder:expr) $name:ident; $($tail:tt)*) => {
        let mut $name: Vec<Option<Register>> = $builder.split_all($name).into_iter().map(|q| Some(q)).collect();
    };
    (@splitter($builder:expr) $name:ident, $($tail:tt)*) => {
        let mut $name: Vec<Option<Register>> = $builder.split_all($name).into_iter().map(|q| Some(q)).collect();
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };

    (@joiner($builder:expr) $name:ident; $($tail:tt)*) => {
        let $name: Vec<Register> = $name.into_iter().map(|q| q.unwrap()).collect();
        let $name: Register = b.merge($name)
    };
    (@joiner($builder:expr) $name:ident, $($tail:tt)*) => {
        let $name: Vec<Register> = $name.into_iter().map(|q| q.unwrap()).collect();
        let $name: Register = b.merge($name)
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };


    (@program_acc($builder:expr, $func:expr, $args:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        // TODO
    };

    (@program_acc($builder:expr, $func:expr, $args:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in $indices {
            tmp_acc.push($name[indx].take().map(|r| Ok(r)).unwrap());
        }
        let tmp_r = $builder.merge(tmp_acc);
        $args.push(tmp_acc);

        program!(@program_acc($builder, $func) $($tail)*)
    };

    (@program($builder:expr) $func:ident $($tail:tt)*) => {
        let mut acc_vec = vec![];
        program!(@program_acc($builder, $func, acc_vec) () <- $($tail)*)
    };

    // (builder; qubits; programs;...)
    ($builder:expr; $($tail:tt)*) => {
        // First reassign each name to a vec
        program!(@splitter($builder) $($tail)*);



        program!(@joiner($builder) $($tail)*);
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
        })?;

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
        })?;
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
        })?;
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
