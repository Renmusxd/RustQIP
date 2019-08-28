/// Common circuits for general usage.
use crate::Register;
use std::iter::Iterator;
use std::ops::{Range, RangeInclusive};

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
/// // Gamma |ra>|rb[2]>
/// // Gamma |ra[0]>|rb>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma ra, rb[2];
///     gamma ra[0], rb;
/// )?;
/// let r = b.merge(vec![ra, rb])?;
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
/// // Gamma |ra[0] ra[2]>|rb[2]>
/// // Gamma |ra>|rb[0] rb[2]>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma |ra[0], ra[2],| rb[2];
///     gamma ra, |rb[0], rb[2],|;
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
///
/// Example with inline ranges:
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
/// // Gamma |ra[0] ra[1]>|ra[2]>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma ra[0..2], ra[2];
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
///
/// Example with inline control:
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
///     let ra = rs.pop().unwrap();
///     let ra = b.not(ra);
///     Ok(vec![ra])
/// };
///
/// // |ra[0] ra[1]> control Gamma |rb[2]>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     control gamma ra[0..2], rb[2];
/// )?;
/// let r = b.merge(vec![ra, rb])?;
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
        let tmp_register_wrapper = RegisterDataWrapper::new(&$name, $reg_vec.len());
        let tmp_name: Vec<Option<Register>> = $builder.split_all($name).into_iter().map(Some).collect();
        let $name = tmp_register_wrapper;
        $reg_vec.push((tmp_name, tmp_indices, stringify!($name)));
    };
    (@splitter($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@splitter($builder, $reg_vec) $name; $($tail)*);
        program!(@splitter($builder, $reg_vec) $($tail)*);
    };

    (@joiner($builder:expr, $reg_vec:ident) $name:ident; $($tail:tt)*) => {
        let $name: Vec<Register> = $reg_vec.pop().unwrap().0.into_iter().map(|q| q.unwrap()).collect();
        let $name: Register = $builder.merge($name)?;
    };
    (@joiner($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@joiner($builder, $reg_vec) $($tail)*);
        program!(@joiner($builder, $reg_vec) $name; $($tail)*);
    };

    // like args_acc for groups
    // no indices
    // closing with semicolon (no indices)
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident,|; $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args, $group_vec) $name tmp_all_indices,|; $($tail)*)
    };
    // closing with comma (no indices)
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident,| $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args, $group_vec) $name tmp_all_indices,| $($tail)*)
    };
    // accumulating (no indices)
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args, $group_vec) $name tmp_all_indices, $($tail)*)
    };
    //opening (must be with comma) (no indices)
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) |$name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args) |$name tmp_all_indices, $($tail)*)
    };
    // with indices
    // closing with semicolon
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr,|; $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_vec, $group_vec) $name $indices);

        let tmp_r = $builder.merge($group_vec)?;
        $args.push(tmp_r);
    };
    // closing with comma
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr,| $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_vec, $group_vec) $name $indices);

        let tmp_r = $builder.merge($group_vec)?;
        $args.push(tmp_r);

        program!(@args_acc($builder, $reg_vec, $args) $($tail)*)
    };
    // accumulating
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_vec, $group_vec) $name $indices);
        program!(@args_acc($builder, $reg_vec, $args, $group_vec) $($tail)*)
    };
    //opening (must be with comma)
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) |$name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_grouped_args: Vec<Register> = vec![];
        program!(@args_acc($builder, $reg_vec, $args, tmp_grouped_args) $name $indices, $($tail)*)
    };
    // @args_acc for cases where no indices are provided
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) $name:ident; $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args) $name tmp_all_indices; $($tail)*)
    };
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) $name:ident, $($tail:tt)*) => {
        let tmp_all_indices: Vec<_> = (0 .. $reg_vec[$name.index].0.len()).collect();
        program!(@args_acc($builder, $reg_vec, $args) $name tmp_all_indices, $($tail)*)
    };
    // Extract the indices from each register and add to the vector of args to be passed to func.
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_vec, $args) $name $indices);
        program!(@args_acc($builder, $reg_vec, $args) $($tail)*)
    };
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) $name:ident $indices:expr;) => {
        program!(@extract_to_args($builder, $reg_vec, $args) $name $indices);
    };
    (@args_acc($builder:expr, $reg_vec:ident, $args:ident) $name:ident $indices:expr; $($tail:tt)*) => {
        program!(@args_acc($builder, $reg_vec, $args) $name $indices;);
    };

    (@extract_to_args($builder:expr, $reg_vec:ident, $args:ident) $name:ident $indices:expr) => {
        let mut tmp_acc: Vec<Register> = vec![];
        for indx in &$indices {
             let tmp_indx_iter: QubitIndices = indx.into();
             for indx in tmp_indx_iter.get_indices() {
                 let tmp_reg = $reg_vec[$name.index].0[indx].take();
                 let tmp_reg = tmp_reg.ok_or_else(|| CircuitError::new(format!("Failed to fetch {}[{}]. May have already been used on this line.", $reg_vec[$name.index].2, indx)))?;
                 tmp_acc.push(tmp_reg);
             }
        }
        if tmp_acc.len() > 0 {
            let tmp_r = $builder.merge(tmp_acc)?;
            $args.push(tmp_r);
        }
    };

    (@replace_registers($builder:expr, $reg_vec:ident, $to_replace:ident)) => {
        // This is not efficient, but the best I can do without fancy custom structs or, more
        // importantly, the ability to make identifiers in macros.
        // Luckily it's O(n^2) where n can't be too big because the actual circuit simulation is
        // O(2^n).
        // for each register in output, for each qubit in the register, find the correct spot in the
        // input register arrays where it originally came from.
        for tmp_register in $to_replace.into_iter() {
            let tmp_indices = tmp_register.indices.clone();
            let tmp_registers = $builder.split_all(tmp_register);
            for (tmp_register, tmp_index) in tmp_registers.into_iter().zip(tmp_indices.into_iter()) {
                for (tmp_reg_holes, tmp_reg_hole_indices, _) in $reg_vec.iter_mut() {
                    let found = tmp_reg_hole_indices.iter()
                        .position(|hole_indx| *hole_indx == tmp_index);
                    if let Some(found) = found {
                        tmp_reg_holes[found] = Some(tmp_register);
                        break;
                    }
                }
            }
        }
        // Now check for abandoned qubits
        for (tmp_regs, tmp_indices, tmp_name) in $reg_vec.iter_mut() {
            tmp_regs.iter().zip(tmp_indices.iter()).enumerate().try_for_each(|(indx, (opt_reg, reg_indx))| {
                if opt_reg.is_none() {
                    let func_name = stringify!($func);
                    CircuitError::make_err(format!("Could not retrieve {}[{}] (absolute index {}). It was not returned from call to {}.", tmp_name, indx, reg_indx, func_name))
                } else {
                    Ok(())
                }
            })?;
        }
    };

    // "End of program list" group
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident;) => {};
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr;) => {};
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident,|;) => {};
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr,|;) => {};

    // "Start of register group" group
    (@skip_to_next_program($builder:expr, $reg_vec:ident) |$name:ident, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_vec:ident) |$name:ident $indices:expr, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };

    // The ",|;" group must be above the ",| (tail)" group
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident,|; $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr,|; $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) $($tail)*)
    };

    // ",| (tail)" group
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident,| $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr,| $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };

    // Execute next program
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident; $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr; $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) $($tail)*)
    };
    // Trivial skips.
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_vec:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*)
    };

    // Start parsing a program of the form "control function [register <indices>, ...];"
    (@program($builder:expr, $reg_vec:ident) control $func:ident $($tail:tt)*) => {
        program!(@program($builder, $reg_vec) control(!0) $func $($tail)*)
    };
    // Start parsing a program of the form "control function [register <indices>, ...];"
    (@program($builder:expr, $reg_vec:ident) control($control:expr) $func:ident $($tail:tt)*) => {
        // Get all args
        let mut tmp_acc_vec: Vec<Register> = vec![];
        program!(@args_acc($builder, $reg_vec, tmp_acc_vec) $($tail)*);
        let tmp_cr = tmp_acc_vec.remove(0);

        let tmp_crs = $builder.split_all(tmp_cr);
        let (tmp_crs,_) = tmp_crs.into_iter().fold((vec![], $control), |(mut qubit_acc, mask_acc), qubit| {
            let lowest = mask_acc & 1;
            let qubit = if lowest == 0 {
                $builder.not(qubit)
            } else {
                qubit
            };
            qubit_acc.push(qubit);
            (qubit_acc, mask_acc >> 1)
        });
        let tmp_cr = $builder.merge(tmp_crs)?;

        let mut tmp_cb = $builder.with_condition(tmp_cr);

        // Now all the args are in acc_vec
        let mut tmp_results: Vec<Register> = $func(&mut tmp_cb, tmp_acc_vec)?;
        let tmp_cr = tmp_cb.release_register();

        let tmp_crs = $builder.split_all(tmp_cr);
        let (tmp_crs,_) = tmp_crs.into_iter().fold((vec![], $control), |(mut qubit_acc, mask_acc), qubit| {
            let lowest = mask_acc & 1;
            let qubit = if lowest == 0 {
                $builder.not(qubit)
            } else {
                qubit
            };
            qubit_acc.push(qubit);
            (qubit_acc, mask_acc >> 1)
        });
        let tmp_cr = $builder.merge(tmp_crs)?;

        tmp_results.push(tmp_cr);
        program!(@replace_registers($builder, $reg_vec, tmp_results));
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*);
    };
    // Start parsing a program of the form "function [register <indices>, ...];"
    (@program($builder:expr, $reg_vec:ident) $func:ident $($tail:tt)*) => {
        // Get all args
        let mut acc_vec: Vec<Register> = vec![];
        program!(@args_acc($builder, $reg_vec, acc_vec) $($tail)*);

        // Now all the args are in acc_vec
        let tmp_results: Vec<Register> = $func($builder, acc_vec)?;
        program!(@replace_registers($builder, $reg_vec, tmp_results));
        program!(@skip_to_next_program($builder, $reg_vec) $($tail)*);
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
            let tmp_f = |b: &mut dyn UnitaryBuilder| {
                // First reassign each name to a vec index
                let mut register_vec: Vec<(Vec<Option<Register>>, Vec<u64>, &str)> = vec![];
                program!(@splitter(b, register_vec) $($tail)*);

                program!(@skip_to_program(b, register_vec) $($tail)*);

                program!(@joiner(b, register_vec) $($tail)*);
                Ok(program!(@name_tuple () <- $($tail)*))
            };
            tmp_f($builder)
        }
    };
}

/// Allows the wrapping of a function with signature:
/// `Fn(&mut dyn UnitaryBuilder, Register, Register, ...) -> (Register, ...)` or
/// `Fn(&mut dyn UnitaryBuilder, Register, Register, ...) -> Result<(Register, ...), CircuitError>`
/// to make a new function with signature:
/// `Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>`
/// and is therefore compatible with `program!`.
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
/// fn gamma(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> (Register, Register) {
///     let (ra, rb) = b.cnot(ra, rb);
///     (ra, rb)
/// }
///
/// wrap_fn!(wrapped_gamma, gamma, ra, rb);
///
/// // Gamma |ra>|rb[2]>
/// // Gamma |ra[0]>|rb>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     wrapped_gamma ra, rb[2];
///     wrapped_gamma ra[0], rb;
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
/// # Example with Result
/// ```
/// use qip::*;
/// # fn main() -> Result<(), CircuitError> {
///
/// let n = 3;
/// let mut b = OpBuilder::new();
/// let ra = b.register(n)?;
/// let rb = b.register(n)?;
///
/// fn gamma(b: &mut dyn UnitaryBuilder, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
///     let (ra, rb) = b.cnot(ra, rb);
///     Ok((ra, rb))
/// }
///
/// wrap_fn!(wrapped_gamma, (gamma), ra, rb);
///
/// // Gamma |ra>|rb[2]>
/// // Gamma |ra[0]>|rb>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     wrapped_gamma ra, rb[2];
///     wrapped_gamma ra[0], rb;
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
///# Example with UnitaryBuilder function
/// ```
/// use qip::*;
/// # fn main() -> Result<(), CircuitError> {
///
/// let n = 3;
/// let mut b = OpBuilder::new();
/// let ra = b.register(n)?;
/// let rb = b.register(n)?;
///
/// wrap_fn!(cnot, UnitaryBuilder::cnot, ra, rb);
///
/// // cnot |ra>|rb[2]>
/// // cnot |ra[0]>|rb>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     cnot ra, rb[2];
///     cnot ra[0], rb;
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! wrap_fn {
    (@names ($($body:tt)*) <- $name:ident) => {
        ($($body)* $name)
    };
    (@names ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        wrap_fn!(@names ($($body)* $name,) <- $($tail)*)
    };
    (@invoke($func:expr, $builder:expr) ($($body:tt)*) <- $name:ident) => {
        $func($builder, $($body)* $name)
    };
    (@invoke($func:expr, $builder:expr) ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        wrap_fn!(@invoke($func, $builder) ($($body)* $name,) <- $($tail)*)
    };
    (@unwrap_regs($func:expr, $rs:ident) $name:ident) => {
        let $name = $rs.pop().ok_or_else(|| CircuitError::new(format!("Error unwrapping {} for {}", stringify!($name), stringify!($func))))?;
    };
    (@unwrap_regs($func:expr, $rs:ident) $name:ident, $($tail:tt)*) => {
        wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
        let $name = $rs.pop().ok_or_else(|| CircuitError::new(format!("Error unwrapping {} for {}", stringify!($name), stringify!($func))))?;
    };
    (@wrap_regs($rs:ident) $name:ident) => {
        $rs.push($name);
    };
    (@wrap_regs($rs:ident) $name:ident, $($tail:tt)*) => {
        $rs.push($name);
        wrap_fn!(@wrap_regs($rs) $($tail)*);
    };
    (@result_body($builder:expr, $func:expr, $rs:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*) ?;
            let mut $rs: Vec<Register> = vec![];
            wrap_fn!(@wrap_regs($rs) $($tail)*);
            Ok($rs)
        }
    };
    (@raw_body($builder:expr, $func:expr, $rs:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*);
            let mut $rs: Vec<Register> = vec![];
            wrap_fn!(@wrap_regs($rs) $($tail)*);
            Ok($rs)
        }
    };
    (pub $newfunc:ident, ($func:expr), $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
            wrap_fn!(@result_body(b, $func, rs) $($tail)*)
        }
    };
    (pub $newfunc:ident, $func:expr, $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs) $($tail)*)
        }
    };
    ($newfunc:ident, ($func:expr), $($tail:tt)*) => {
        fn $newfunc(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
            wrap_fn!(@result_body(b, $func, rs) $($tail)*)
        }
    };
    ($newfunc:ident, $func:expr, $($tail:tt)*) => {
        fn $newfunc(b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>) -> Result<Vec<Register>, CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs) $($tail)*)
        }
    };
}

/// Helper struct for macro iteration with usize, ranges, or vecs
#[derive(Debug)]
pub struct QubitIndices {
    indices: Vec<usize>,
}

impl QubitIndices {
    /// Get indices from struct.
    pub fn get_indices(self) -> Vec<usize> {
        self.indices
    }
}

impl From<Range<usize>> for QubitIndices {
    fn from(indices: Range<usize>) -> Self {
        let indices: Vec<_> = indices.collect();
        indices.into()
    }
}
impl From<Range<u64>> for QubitIndices {
    fn from(indices: Range<u64>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<Range<u32>> for QubitIndices {
    fn from(indices: Range<u32>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<Range<i64>> for QubitIndices {
    fn from(indices: Range<i64>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<Range<i32>> for QubitIndices {
    fn from(indices: Range<i32>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<&Range<usize>> for QubitIndices {
    fn from(indices: &Range<usize>) -> Self {
        (indices.start..indices.end).into()
    }
}
impl From<&Range<u64>> for QubitIndices {
    fn from(indices: &Range<u64>) -> Self {
        (indices.start..indices.end).into()
    }
}
impl From<&Range<u32>> for QubitIndices {
    fn from(indices: &Range<u32>) -> Self {
        (indices.start..indices.end).into()
    }
}
impl From<&Range<i64>> for QubitIndices {
    fn from(indices: &Range<i64>) -> Self {
        (indices.start..indices.end).into()
    }
}
impl From<&Range<i32>> for QubitIndices {
    fn from(indices: &Range<i32>) -> Self {
        (indices.start..indices.end).into()
    }
}
impl From<RangeInclusive<usize>> for QubitIndices {
    fn from(indices: RangeInclusive<usize>) -> Self {
        let indices: Vec<_> = indices.collect();
        indices.into()
    }
}
impl From<RangeInclusive<u64>> for QubitIndices {
    fn from(indices: RangeInclusive<u64>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<RangeInclusive<u32>> for QubitIndices {
    fn from(indices: RangeInclusive<u32>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<RangeInclusive<i64>> for QubitIndices {
    fn from(indices: RangeInclusive<i64>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<RangeInclusive<i32>> for QubitIndices {
    fn from(indices: RangeInclusive<i32>) -> Self {
        let indices: Vec<_> = indices.map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<&RangeInclusive<usize>> for QubitIndices {
    fn from(indices: &RangeInclusive<usize>) -> Self {
        let indices: Vec<_> = indices.clone().collect();
        indices.into()
    }
}
impl From<&RangeInclusive<u64>> for QubitIndices {
    fn from(indices: &RangeInclusive<u64>) -> Self {
        let indices: Vec<_> = indices.clone().map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<&RangeInclusive<u32>> for QubitIndices {
    fn from(indices: &RangeInclusive<u32>) -> Self {
        let indices: Vec<_> = indices.clone().map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<&RangeInclusive<i64>> for QubitIndices {
    fn from(indices: &RangeInclusive<i64>) -> Self {
        let indices: Vec<_> = indices.clone().map(|indx| indx as usize).collect();
        indices.into()
    }
}
impl From<&RangeInclusive<i32>> for QubitIndices {
    fn from(indices: &RangeInclusive<i32>) -> Self {
        let indices: Vec<_> = indices.clone().map(|indx| indx as usize).collect();
        indices.into()
    }
}

impl<T: Into<usize>> From<Vec<T>> for QubitIndices {
    fn from(indices: Vec<T>) -> Self {
        let indices: Vec<usize> = indices.into_iter().map(|indx| indx.into()).collect();
        Self { indices }
    }
}

impl From<usize> for QubitIndices {
    fn from(item: usize) -> Self {
        Self {
            indices: vec![item],
        }
    }
}
impl From<&usize> for QubitIndices {
    fn from(item: &usize) -> Self {
        (*item).into()
    }
}

impl From<u64> for QubitIndices {
    fn from(item: u64) -> Self {
        Self {
            indices: vec![item as usize],
        }
    }
}
impl From<&u64> for QubitIndices {
    fn from(item: &u64) -> Self {
        (*item).into()
    }
}

impl From<u32> for QubitIndices {
    fn from(item: u32) -> Self {
        Self {
            indices: vec![item as usize],
        }
    }
}
impl From<&u32> for QubitIndices {
    fn from(item: &u32) -> Self {
        (*item).into()
    }
}

impl From<i64> for QubitIndices {
    fn from(item: i64) -> Self {
        Self {
            indices: vec![item as usize],
        }
    }
}
impl From<&i64> for QubitIndices {
    fn from(item: &i64) -> Self {
        (*item).into()
    }
}

impl From<i32> for QubitIndices {
    fn from(item: i32) -> Self {
        Self {
            indices: vec![item as usize],
        }
    }
}
impl From<&i32> for QubitIndices {
    fn from(item: &i32) -> Self {
        (*item).into()
    }
}

/// A struct which wraps the metadata for a Register, this is so that expressions which reference
/// the register can still be used inside the program! macro.
#[derive(Debug)]
pub struct RegisterDataWrapper {
    /// Indices of the register
    pub indices: Vec<u64>,
    /// Number of qubits in the register
    pub n: u64,
    /// Index of the register as arg to program!
    pub index: usize,
}

impl RegisterDataWrapper {
    /// Make a new RegisterDataWrapper for a register (with given index)
    pub fn new(r: &Register, index: usize) -> Self {
        Self {
            indices: r.indices.clone(),
            n: r.n(),
            index,
        }
    }

    /// Number of qubits in the register
    pub fn n(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
mod common_circuit_tests {
    use super::*;
    use crate::pipeline::make_circuit_matrix;
    use crate::{run_debug, CircuitError, OpBuilder, Register, UnitaryBuilder};

    #[test]
    fn test_program_macro() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder,
                    mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let (ra, rb) = program!(&mut b, ra, rb;
            cnot ra, rb[2];
            cnot ra[0], rb;
        )?;
        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (r, rb) = b.split(rb, &[2])?;
        let (ra, r) = b.cnot(ra, r);
        let rb = b.merge_with_indices(rb, vec![r], &[2])?;
        let (r, ra) = b.split(ra, &[0])?;
        let (r, rb) = b.cnot(r, rb);
        let ra = b.merge_with_indices(ra, vec![r], &[0])?;
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_reuse() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder,
                    mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let r = program!(&mut b, r;
            cnot r[0], r[1];
        )?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (r1, r2) = b.split(r, &[0])?;
        let (r2, r3) = b.split(r2, &[0])?;
        let (r1, r2) = b.cnot(r1, r2);
        let r = b.merge(vec![r1, r2, r3])?;
        run_debug(&r)?;
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

        let cnot = |b: &mut dyn UnitaryBuilder,
                    mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let (ra, rb) = program!(&mut b, ra, rb;
            cnot |ra[0], ra[2],| rb[2];
        )?;

        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (r, rb) = b.split(rb, &[2])?;
        let (ra_side, ra_mid) = b.split(ra, &[0, 2])?;
        let (ra_side, r) = b.cnot(ra_side, r);
        let ra_sides = b.split_all(ra_side);
        let ra = b.merge_with_indices(ra_mid, ra_sides, &[0, 2])?;
        let rb = b.merge_with_indices(rb, vec![r], &[2])?;
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2 * n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_inline_range() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let cnot = |b: &mut dyn UnitaryBuilder,
                    mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

        let r = program!(&mut b, r;
            cnot r[0..2], r[2];
        )?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (r1, r2) = b.split(r, &[2])?;
        let (r2, r1) = b.cnot(r2, r1);
        let r = b.merge(vec![r1, r2])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_inline_control() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let not = |b: &mut dyn UnitaryBuilder,
                   mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let ra = rs.pop().unwrap();
            let ra = b.not(ra);
            Ok(vec![ra])
        };

        let r = program!(&mut b, r;
            control not r[0..2], r[2];
        )?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (r1, r2) = b.split(r, &[2])?;
        let (r2, r1) = b.cnot(r2, r1);
        let r = b.merge(vec![r1, r2])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_inline_control_expr() -> Result<(), CircuitError> {
        let n = 3;

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let not = |b: &mut dyn UnitaryBuilder,
                   mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let ra = rs.pop().unwrap();
            let ra = b.not(ra);
            Ok(vec![ra])
        };

        let r = program!(&mut b, r;
            control(00) not r[0..2], r[2];
        )?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (r1, r2) = b.split(r, &[2])?;
        let r2 = b.not(r2);
        let (r2, r1) = b.cnot(r2, r1);
        let r2 = b.not(r2);
        let r = b.merge(vec![r1, r2])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_repeated() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let ra = b.qubit();
        let rb = b.qubit();

        let not = |b: &mut dyn UnitaryBuilder,
                   mut rs: Vec<Register>|
         -> Result<Vec<Register>, CircuitError> {
            let ra = rs.pop().unwrap();
            let ra = b.not(ra);
            Ok(vec![ra])
        };

        let (ra, rb) = program!(&mut b, ra, rb;
            control not ra, rb;
        )?;

        let (ra, rb) = program!(&mut b, ra, rb;
            control not ra, rb;
        )?;
        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.qubit();
        let rb = b.qubit();
        let (ra, rb) = b.cnot(ra, rb);
        let (ra, rb) = b.cnot(ra, rb);
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    fn simple_fn(b: &mut dyn UnitaryBuilder, ra: Register) -> Register {
        b.not(ra)
    }

    #[test]
    fn wrap_simple_fn() -> Result<(), CircuitError> {
        let n = 1;
        wrap_fn!(wrapped_simple_fn, simple_fn, ra);

        let mut b = OpBuilder::new();
        let r = b.register(n)?;

        let r = program!(&mut b, r;
            wrapped_simple_fn r;
        )?;

        run_debug(&r)?;
        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let r = simple_fn(&mut b, r);
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    fn complex_fn(
        b: &mut dyn UnitaryBuilder,
        ra: Register,
        rb: Register,
        rc: Register,
    ) -> Result<(Register, Register, Register), CircuitError> {
        let mut cb = b.with_condition(ra);
        let (rb, rc) = cb.cnot(rb, rc);
        let ra = cb.release_register();
        Ok((ra, rb, rc))
    }

    #[test]
    fn wrap_complex_fn() -> Result<(), CircuitError> {
        let n = 1;
        wrap_fn!(wrapped_complex_fn, (complex_fn), ra, rb, rc);

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let rc = b.register(n)?;

        let (ra, rb, rc) = program!(&mut b, ra, rb, rc;
            wrapped_complex_fn ra, rb, rc;
        )?;
        let r = b.merge(vec![ra, rb, rc])?;

        run_debug(&r)?;
        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let rc = b.register(n)?;
        let (ra, rb, rc) = complex_fn(&mut b, ra, rb, rc)?;
        let r = b.merge(vec![ra, rb, rc])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn wrap_unitary_fn() -> Result<(), CircuitError> {
        let n = 1;
        wrap_fn!(wrapped_cnot, UnitaryBuilder::cnot, ra, rb);

        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;

        let (ra, rb) = program!(&mut b, ra, rb;
            wrapped_cnot ra, rb;
        )?;
        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;
        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (ra, rb) = b.cnot(ra, rb);
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }
}
