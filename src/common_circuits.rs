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
        let tmp: Vec<Register> = $builder.split_all($name);
        let $name: Register = $builder.merge_with_indices($remaining.pop().unwrap(), tmp, &$indices)?;
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

/// A helper macro for applying functions to specific qubits in registers.
///
/// The macro takes an expression which yields `&mut dyn UnitaryBuilder`, a list of registers, then
/// a series of function call expressions of the form:
/// `function [register <indices?>, ...];`
///
/// Registers can be groups by surrounding them with vertical bars, for example:
/// `function |ra, rb[0,2],| rc`
/// (Notice the comma is inside the bars, not the best syntax but macros are restrictive).
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
///
/// let gamma = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
///     let rb = rs.pop().unwrap();
///     let ra = rs.pop().unwrap();
///     let (ra, rb) = b.cnot(ra, rb);
///     Ok(vec![ra, rb])
/// };
///
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma ra, rb[2];
///     gamma ra[0], rb;
/// );
/// let r = b.merge(vec![ra, rb]);
///
/// # Ok(())
/// # }
/// ```
///
/// Example with grouping, `ra[0]` and `ra[2]` are selected but `ra[1]` is not.
/// ```
/// use qip::*;
/// # fn main() -> Result<(), CircuitError> {
///
/// let n = 3;
/// let mut b = OpBuilder::new();
/// let ra = b.register(n)?;
/// let rb = b.register(n)?;
///
/// let gamma = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
///     let rb = rs.pop().unwrap();
///     let ra = rs.pop().unwrap();
///     let (ra, rb) = b.cnot(ra, rb);
///     Ok(vec![ra, rb])
/// };
///
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma |ra[0], ra[2],| rb[2];
/// );
/// let r = b.merge(vec![ra, rb]);
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! program {
    (@name_tuple ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@name_tuple ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        program!(@name_tuple ($($body)* $name,) <- $($tail)*)
    };

    (@splitter($builder:expr, $reg_vec:ident) $name:ident; $($tail:tt)*) => {
        let tmp_indices = $name.indices.clone();
        let tmp_name: Vec<Option<Register>> = $builder.split_all($name).into_iter().map(|q| Some(q)).collect();
        let $name = $reg_vec.len();
        $reg_vec.push((tmp_name, tmp_indices));
    };
    (@splitter($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@splitter($builder, $reg_vec) $name; $($tail)*)
        program!(@splitter($builder, $reg_vec) $($tail)*)
    };

    (@joiner($builder:expr, $reg_vec:ident) $name:ident; $($tail:tt)*) => {
        let $name: Vec<Register> = $reg_vec.pop().unwrap().0.into_iter().map(|q| q.unwrap()).collect();
        let $name: Register = $builder.merge($name)
    };
    (@joiner($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@joiner($builder, $reg_vec) $name; $($tail)*)
        program!(@joiner($builder, $reg_vec) $($tail)*)
    };

    // like program_acc for parenthesis
    // no indices
    // closing with semicolon (no indices)
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident;| $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args, $group_vec) $name tmp_all_indices;| $($tail)*)
    };
    // closing with comma (no indices)
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident,| $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args, $group_vec) $name tmp_all_indices,| $($tail)*)
    };
    // accumulating (no indices)
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args, $group_vec) $name tmp_all_indices, $($tail)*)
    };
    //opening (must be with comma) (no indices)
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) |$name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args, $group_vec) |$name tmp_all_indices, $($tail)*)
    };
    // with indices
    // closing with semicolon
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident $indices:expr;| $($tail:tt)*) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in &$indices {
            tmp_acc.push($reg_vec[$name].0[*indx].take().unwrap());
        }
        let tmp_r = $builder.merge(tmp_acc);
        $group_vec.push(tmp_r);
        let tmp_r = $builder.merge($group_vec);
        $args.push(tmp_r);

        program!(@program_execute($builder, $reg_vec, $func, $args))
    };
    // closing with comma
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident $indices:expr,| $($tail:tt)*) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in &$indices {
            tmp_acc.push($reg_vec[$name].0[*indx].take().unwrap());
        }
        let tmp_r = $builder.merge(tmp_acc);
        $group_vec.push(tmp_r);
        let tmp_r = $builder.merge($group_vec);
        $args.push(tmp_r);

        program!(@program_acc($builder, $reg_vec, $func, $args) $($tail)*)
    };
    // accumulating
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident, $group_vec:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in &$indices {
            tmp_acc.push($reg_vec[$name].0[*indx].take().unwrap());
        }
        let tmp_r = $builder.merge(tmp_acc);
        $group_vec.push(tmp_r);
        program!(@program_acc($builder, $reg_vec, $func, $args, $group_vec) $($tail)*)
    };
    //opening (must be with comma)
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) |$name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_grouped_args: Vec<Register> = vec![];
        program!(@program_acc($builder, $reg_vec, $func, $args, tmp_grouped_args) $name $indices, $($tail)*)
    };


    // @program_acc for cases where no indices are provided
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) $name:ident; $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args) $name tmp_all_indices; $($tail)*)
    };
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) $name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name].0.len()).collect();
        program!(@program_acc($builder, $reg_vec, $func, $args) $name tmp_all_indices, $($tail)*)
    };
    // Extract the indices from each register and add to the vector of args to be passed to func.
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in &$indices {
            tmp_acc.push($reg_vec[$name].0[*indx].take().unwrap());
        }
        let tmp_r = $builder.merge(tmp_acc);
        $args.push(tmp_r);

        program!(@program_acc($builder, $reg_vec, $func, $args) $($tail)*)
    };
    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) $name:ident $indices:expr;) => {
        let mut tmp_acc: Vec < Register > = vec ! [];
        for indx in & $ indices {
        tmp_acc.push( $ reg_vec[ $ name].0[ * indx].take().unwrap());
        }
        let tmp_r = $ builder.merge(tmp_acc);
        $args.push(tmp_r);
        program!(@program_execute($builder, $reg_vec, $func, $args))
    };

    (@program_execute($builder:expr, $reg_vec:ident, $func:expr, $args:ident)) => {
        let tmp_results: Vec<Register> = $func($builder, $args)?;
        // This is not efficient, but the best I can do without fancy custom structs or, more
        // importantly, the ability to make identifiers in macros.
        // Luckily it's O(n^2) where n can't be too big because the actual circuit simulation is
        // O(2^n).
        // for each register in output, for each qubit in the register, find the correct spot in the
        // input register arrays where it originally came from.
        for tmp_register in tmp_results.into_iter() {
            let tmp_indices = tmp_register.indices.clone();
            let tmp_registers = $builder.split_all(tmp_register);
            for (tmp_register, tmp_index) in tmp_registers.into_iter().zip(tmp_indices.into_iter()) {
                for (tmp_reg_holes, tmp_reg_hole_indices) in $reg_vec.iter_mut() {
                    let found = tmp_reg_hole_indices.iter()
                        .position(|hole_indx| *hole_indx == tmp_index);
                    if let Some(found) = found {
                        tmp_reg_holes[found] = Some(tmp_register);
                        break;
                    }
                }
            }
        }
    };

    (@program_acc($builder:expr, $reg_vec:ident, $func:expr, $args:ident) $name:ident $indices:expr; $($tail:tt)*) => {
        program!(@program_acc($builder, $reg_vec, $func, $args) $name $indices;);
        program!(@program($builder, $reg_vec) $($tail)*);
    };

    // Start parsing a program of the form "function [register <indices>, ...];"
    (@program($builder:expr, $reg_vec:ident) $func:ident $($tail:tt)*) => {
        let mut acc_vec: Vec<Register> = vec![];
        program!(@program_acc($builder, $reg_vec, $func, acc_vec) $($tail)*)
    };

    // Skip past the register list and start running programs.
    (@skip_to_program($builder:expr, $reg_vec:ident) $name:ident; $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_program($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@skip_to_program($builder, $reg_vec) $($tail)*)
    };

    // (builder, register_1, ...; programs; ...)
    ($builder:expr, $($tail:tt)*) => {
        {
            // First reassign each name to a vec index
            let mut register_vec: Vec<(Vec<Option<Register>>, Vec<u64>)> = vec![];
            program!(@splitter($builder, register_vec) $($tail)*);

            program!(@skip_to_program($builder, register_vec) $($tail)*);

            program!(@joiner($builder, register_vec) $($tail)*);
            program!(@name_tuple () <- $($tail)*)
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

    #[test]
    fn test_program_macro() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let (ra, rb) = program!(&mut b, ra, rb;
            cnot ra, rb[2];
            cnot ra[0], rb;
        );
        let r = b.merge(vec![ra, rb]);

        run_debug(&r);

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (r, rb) = b.split(rb, &[2])?;
        let (ra, r) = b.cnot(ra, r);
        let rb = b.merge_with_indices(rb, vec![r], &[2])?;
        let (r, ra) = b.split(ra, &[0])?;
        let (r, rb) = b.cnot(r, rb);
        let ra = b.merge_with_indices(ra, vec![r], &[0])?;
        let r = b.merge(vec![ra, rb]);
        run_debug(&r);
        let basic_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }


    #[test]
    fn test_program_macro_reuse() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let r = program!(&mut b, r;
            cnot r[0], r[1];
        );

        run_debug(&r);

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (r1, r2) = b.split(r, &[0])?;
        let (r2, r3) = b.split(r2, &[0])?;
        let (r1, r2) = b.cnot(r1, r2);
        let r = b.merge(vec![r1, r2, r3]);
        run_debug(&r);
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_group() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let (ra, rb) = program!(&mut b, ra, rb;
            cnot |ra[0], ra[2],| rb[2];
        );
        let r = b.merge(vec![ra, rb]);

        run_debug(&r);

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (r, rb) = b.split(rb, &[2])?;
        let (ra_side, ra_mid) = b.split(ra, &[0, 2])?;
        let (ra_side, r) = b.cnot(ra_side, r);
        let ra_sides = b.split_all(ra_side);
        let ra = b.merge_with_indices(ra_mid, ra_sides, &[0, 2])?;
        let rb = b.merge_with_indices(rb, vec![r], &[2])?;
        let r = b.merge(vec![ra, rb]);
        run_debug(&r);
        let basic_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }
}
