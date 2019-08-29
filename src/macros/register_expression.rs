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
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! register_expr {
    // Split the names into the selected indices, pushing the remaining to the acc vector.
    (@splitter($acc:ident) $builder:expr, $name:ident $indices:expr; $($tail:tt)*) => {
        let $name: (Register, Option<Register>) = $builder.split($name, &$indices)?;
        $acc.push($name.1);
        let $name = $name.0;
    };
    (@splitter($acc:ident) $builder:expr, $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@splitter($acc) $builder, $name $indices; $($tail)*);
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };
    (@splitter($acc:ident) $builder:expr, $name:ident; $($tail:tt)*) => {};
    (@splitter($acc:ident) $builder:expr, $name:ident, $($tail:tt)*) => {
        register_expr!(@splitter($acc) $builder, $($tail)*)
    };

    // Join together all the names back to their original indices by pulling from the
    // remaining vector.
    (@joiner($remaining:ident) $builder:expr, $name:ident $indices:expr; $($tail:tt)*) => {
        let $name: Register = if let Some(tmp_r) = $remaining.pop().unwrap() {
            let tmp: Vec<Register> = $builder.split_all($name);
            $builder.merge_with_indices(tmp_r, tmp, &$indices)?
        } else {
            $name
        };
    };
    (@joiner($remaining:ident) $builder:expr, $name:ident $indices:expr, $($tail:tt)*) => {
        register_expr!(@joiner($remaining) $builder, $($tail)*);
        register_expr!(@joiner($remaining) $builder, $name $indices; $($tail)*);
    };
    (@joiner($remaining:ident) $builder:expr, $name:ident; $($tail:tt)*) => {};
    (@joiner($remaining:ident) $builder:expr, $name:ident, $($tail:tt)*) => {
        register_expr!(@joiner($remaining) $builder, $($tail)*);
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
        register_expr!(@func($builder, $args) ($($body)*) <- $name; $func)
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
                let mut remaining_qubits: Vec<Option<Register>> = vec![];
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

#[cfg(test)]
mod common_circuit_tests {
    use super::*;
    use crate::pipeline::make_circuit_matrix;
    use crate::{run_debug, CircuitError, OpBuilder, Register, UnitaryBuilder};

    #[test]
    fn test_work_on_macro() -> Result<(), CircuitError> {
        let n = 2;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let r = register_expr!(&mut b, r[0]; |b, r| {
            let r = b.not(r);
            Ok(r)
        })?;
        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (t_r, r) = b.split(r, &[0])?;
        let t_r = b.not(t_r);
        let r = b.merge_with_indices(r.unwrap(), vec![t_r], &[0])?;
        run_debug(&r)?;
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
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (t_rb, rb) = b.split(rb, &[1])?;
        let t_rb = b.not(t_rb);
        let rb = b.merge_with_indices(rb.unwrap(), vec![t_rb], &[1])?;
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
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
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let rb = b.not(rb);
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }
}
