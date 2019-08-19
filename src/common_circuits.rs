use crate::errors::CircuitError;
/// Common circuits for general usage.
use crate::{OpBuilder, Register, UnitaryBuilder};
use std::ops::Range;
use std::iter::Iterator;

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
/// let r = b.merge(vec![ra, rb])?;
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

/// Helper struct for macro iteration with usize, ranges, or vecs
#[derive(Debug)]
pub struct QubitIndices {
    indices: Vec<usize>
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
        (indices.start .. indices.end).into()
    }
}
impl From<&Range<u64>> for QubitIndices {
    fn from(indices: &Range<u64>) -> Self {
        (indices.start .. indices.end).into()
    }
}
impl From<&Range<u32>> for QubitIndices {
    fn from(indices: &Range<u32>) -> Self {
        (indices.start .. indices.end).into()
    }
}
impl From<&Range<i64>> for QubitIndices {
    fn from(indices: &Range<i64>) -> Self {
        (indices.start .. indices.end).into()
    }
}
impl From<&Range<i32>> for QubitIndices {
    fn from(indices: &Range<i32>) -> Self {
        (indices.start .. indices.end).into()
    }
}

impl<T: Into<usize>> From<Vec<T>> for QubitIndices {
    fn from(indices: Vec<T>) -> Self {
        let indices: Vec<usize> = indices.into_iter().map(|indx| indx.into()).collect();
        Self {
            indices
        }
    }
}

impl From<usize> for QubitIndices {
    fn from(item: usize) -> Self {
        Self {
            indices: vec![item]
        }
    }
}
impl From<&usize> for QubitIndices {
    fn from(item: &usize) -> Self {
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
    pub(crate) index: usize,
}

impl RegisterDataWrapper {
    /// Make a new RegisterDataWrapper for a register (with given index)
    pub fn new(r: &Register, index: usize) -> Self {
        Self {
            indices: r.indices.clone(),
            n: r.n(),
            index
        }
    }

    /// Number of qubits in the register
    pub fn n(&self) -> u64 {
        self.n
    }
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
/// // Gamma |ra>|rb[2]>
/// // Gamma |ra[0]>|rb>
/// let (ra, rb) = program!(&mut b, ra, rb;
///     gamma ra, rb[2];
///     gamma ra[0], rb;
/// );
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
/// );
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
/// );
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
/// );
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
        let tmp_name: Vec<Option<Register>> = $builder.split_all($name).into_iter().map(|q| Some(q)).collect();
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
        program!(@joiner($builder, $reg_vec) $name; $($tail)*);
        program!(@joiner($builder, $reg_vec) $($tail)*);
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
        // Get all args
        let mut tmp_acc_vec: Vec<Register> = vec![];
        program!(@args_acc($builder, $reg_vec, tmp_acc_vec) $($tail)*);
        let tmp_cr = tmp_acc_vec.remove(0);
        let mut tmp_cb = $builder.with_condition(tmp_cr);

        // Now all the args are in acc_vec
        let mut tmp_results: Vec<Register> = $func(&mut tmp_cb, tmp_acc_vec)?;
        let tmp_cr = tmp_cb.release_register();
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
            // First reassign each name to a vec index
            let mut register_vec: Vec<(Vec<Option<Register>>, Vec<u64>, &str)> = vec![];
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
    let ra = b.merge(all_rs).unwrap();
    let rb = b.merge(back_rs).unwrap();

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
        let r = work_on(&mut b, r, &[0], |_b, qs| Ok(qs))?;
        run_debug(&r)?;

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
            Ok(r)
        })?;
        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(n, &r, true);
        let mut b = OpBuilder::new();
        let r = b.register(n)?;
        let (t_r, r) = b.split(r, &[0])?;
        let t_r = b.not(t_r);
        let r = b.merge_with_indices(r, vec![t_r], &[0])?;
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
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let (t_rb, rb) = b.split(rb, &[1])?;
        let t_rb = b.not(t_rb);
        let rb = b.merge_with_indices(rb, vec![t_rb], &[1])?;
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
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
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;

        // Compare to expected value
        let macro_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        let mut b = OpBuilder::new();
        let ra = b.register(n)?;
        let rb = b.register(n)?;
        let rb = b.not(rb);
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
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
        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;

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
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
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

        let cnot = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
            let rb = rs.pop().unwrap();
            let ra = rs.pop().unwrap();
            let (ra, rb) = b.cnot(ra, rb);
            Ok(vec![ra, rb])
        };

//        trace_macros!(true);

        let (ra, rb) = program!(&mut b, ra, rb;
            cnot |ra[0], ra[2],| rb[2];
        );

//        trace_macros!(false);

        let r = b.merge(vec![ra, rb])?;

        run_debug(&r)?;

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
        let r = b.merge(vec![ra, rb])?;
        run_debug(&r)?;
        let basic_circuit = make_circuit_matrix::<f64>(2*n, &r, true);
        assert_eq!(macro_circuit, basic_circuit);
        Ok(())
    }

    #[test]
    fn test_program_macro_inline_range() -> Result<(), CircuitError> {
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
            cnot r[0..2], r[2];
        );

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

        let not = |b: &mut dyn UnitaryBuilder, mut rs: Vec<Register>| -> Result<Vec<Register>, CircuitError> {
            let ra = rs.pop().unwrap();
            let ra = b.not(ra);
            Ok(vec![ra])
        };

        let r = program!(&mut b, r;
            control not r[0..2], r[2];
        );

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
}
