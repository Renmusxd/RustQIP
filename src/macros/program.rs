use crate::errors::CircuitResult;
use crate::prelude::{CircuitBuilder, CircuitError, CliffordTBuilder, QubitRegister};
use crate::Precision;
use std::ops::{Range, RangeInclusive};

#[macro_export]
macro_rules! program {
    (@name_tuple () <- $name:ident; $($tail:tt)*) => {
        $name
    };
    (@name_tuple ($($body:tt)*) <- $name:ident; $($tail:tt)*) => {
        ($($body)* $name)
    };
    (@name_tuple ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        program!(@name_tuple ($($body)* $name,) <- $($tail)*)
    };

    (@splitter($builder:expr, $reg_man:ident) $name:ident; $($tail:tt)*) => {
        let $name = $reg_man.add_register($builder, $name, stringify!($name));
    };
    (@splitter($builder:expr, $reg_man:ident) $name:ident, $($tail:tt)*) => {
        program!(@splitter($builder, $reg_man) $name; $($tail)*);
        program!(@splitter($builder, $reg_man) $($tail)*);
    };

    (@joiner($builder:expr, $reg_man:ident) $name:ident; $($tail:tt)*) => {
        let $name = $reg_man.pop_register($builder);
    };
    (@joiner($builder:expr, $reg_man:ident) $name:ident, $($tail:tt)*) => {
        program!(@joiner($builder, $reg_man) $($tail)*);
        program!(@joiner($builder, $reg_man) $name; $($tail)*);
    };

    // like args_acc for groups
    // no indices
    // closing with semicolon (no indices)
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident,|; $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name);
        $builder.merge_registers($group_vec).map(|r| {
            $args.push(r)
        });
    };
    // closing with comma (no indices)
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident,| $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name);
        $builder.merge_registers($group_vec).map(|r| {
            $args.push(r)
        });
        program!(@args_acc($builder, $reg_man, $args) $($tail)*)
    };
    // accumulating (no indices)
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name);
        program!(@args_acc($builder, $reg_man, $args, $group_vec) $($tail)*)
    };
    //opening (must be with comma) (no indices)
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) |$name:ident, $($tail:tt)*) => {
        let mut tmp_grouped_args: Vec<_> = Vec::default();
        program!(@args_acc($builder, $reg_man, $args, tmp_grouped_args) $name, $($tail)*)
    };
    // with indices
    // closing with semicolon
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr,|; $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name $indices);
        $builder.merge_registers($group_vec).map(|r| {
            $args.push(r)
        });
    };
    // closing with comma
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr,| $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name $indices);
        $builder.merge_registers($group_vec).map(|r| {
            $args.push(r)
        });
        program!(@args_acc($builder, $reg_man, $args) $($tail)*)
    };
    // accumulating
    (@args_acc($builder:expr, $reg_man:ident, $args:ident, $group_vec:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $group_vec) $name $indices);
        program!(@args_acc($builder, $reg_man, $args, $group_vec) $($tail)*)
    };
    //opening (must be with comma)
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) |$name:ident $indices:expr, $($tail:tt)*) => {
        let mut tmp_grouped_args: Vec<_> = Vec::default();
        program!(@args_acc($builder, $reg_man, $args, tmp_grouped_args) $name $indices, $($tail)*)
    };
    // @args_acc for cases where no indices are provided
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) $name:ident; $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $args) $name)
    };
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) $name:ident, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $args) $name);
        program!(@args_acc($builder, $reg_man, $args) $($tail)*)
    };
    // Extract the indices from each register and add to the vector of args to be passed to func.
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@extract_to_args($builder, $reg_man, $args) $name $indices);
        program!(@args_acc($builder, $reg_man, $args) $($tail)*);
    };
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) $name:ident $indices:expr;) => {
        program!(@extract_to_args($builder, $reg_man, $args) $name $indices);
    };
    (@args_acc($builder:expr, $reg_man:ident, $args:ident) $name:ident $indices:expr; $($tail:tt)*) => {
        program!(@args_acc($builder, $reg_man, $args) $name $indices;);
    };

    (@extract_to_args($builder:expr, $reg_man:ident, $args:ident) $name:ident $indices:expr) => {
        let tmp_r = $reg_man.get_registers($builder, $name.index, &$indices)?;
        $args.push(tmp_r);
    };
    (@extract_to_args($builder:expr, $reg_man:ident, $args:ident) $name:ident) => {
        let tmp_r = $reg_man.get_full_register($builder, $name.index)?;
        $args.push(tmp_r);
    };

    (@replace_registers($builder:expr, $reg_man:ident, $to_replace:ident)) => {
        $reg_man.return_registers($builder, $to_replace);
    };

    // "End of program list" group
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident;) => {};
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr;) => {};
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident,|;) => {};
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr,|;) => {};

    // "Start of register group" group
    (@skip_to_next_program($builder:expr, $reg_man:ident) |$name:ident, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_man:ident) |$name:ident $indices:expr, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };

    // The ",|;" group must be above the ",| (tail)" group
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident,|; $($tail:tt)*) => {
        program!(@program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr,|; $($tail:tt)*) => {
        program!(@program($builder, $reg_man) $($tail)*)
    };

    // ",| (tail)" group
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident,| $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr,| $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };

    // Execute next program
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident; $($tail:tt)*) => {
        program!(@program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr; $($tail:tt)*) => {
        program!(@program($builder, $reg_man) $($tail)*)
    };
    // Trivial skips.
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_next_program($builder:expr, $reg_man:ident) $name:ident $indices:expr, $($tail:tt)*) => {
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*)
    };

    // Start parsing a program of the form "control function(arg) [register <indices>, ...];"
    (@program($builder:expr, $reg_man:ident) control $func:ident($funcargs:expr) $($tail:tt)*) => {
        program!(@program($builder, $reg_man) control(!0) $func($funcargs) $($tail)*)
    };
    // Start parsing a program of the form "control function(arg) [register <indices>, ...];"
    (@program($builder:expr, $program_acc:ident) control($control:expr) $func:ident($funcargs:expr) $($tail:tt)*) => {
        let $program_acc = $program_acc.and_then(|mut reg_man| -> Result<$crate::macros::program::RegisterManager<_>, $crate::errors::CircuitError> {
            // Get all args
            let mut tmp_acc_vec: Vec<_> = Vec::default();
            program!(@args_acc($builder, reg_man, tmp_acc_vec) $($tail)*);
            let tmp_cr = tmp_acc_vec.remove(0);

            let tmp_cr = $crate::macros::program::negate_bitmask($builder, tmp_cr, $control);
            let mut tmp_cb = $builder.condition_with(tmp_cr);

            // Now all the args are in acc_vec
            let mut tmp_results: Vec<_> = $func(&mut tmp_cb, tmp_acc_vec, $funcargs)?;

            let tmp_cr = tmp_cb.dissolve();
            let tmp_cr = $crate::macros::program::negate_bitmask($builder, tmp_cr, $control);

            tmp_results.push(tmp_cr);
            program!(@replace_registers($builder, reg_man, tmp_results));
            Ok(reg_man)
        });
        program!(@skip_to_next_program($builder, $reg_man) $($tail)*);
    };

    // Start parsing a program of the form "control function [register <indices>, ...];"
    (@program($builder:expr, $program_acc:ident) control $func:ident $($tail:tt)*) => {
        program!(@program($builder, $program_acc) control(!0) $func $($tail)*)
    };
    // Start parsing a program of the form "control function [register <indices>, ...];"
    (@program($builder:expr, $program_acc:ident) control($control:expr) $func:ident $($tail:tt)*) => {
        let $program_acc = $program_acc.and_then(|mut reg_man| -> Result<$crate::macros::program::RegisterManager<_>, $crate::errors::CircuitError> {
            // Get all args
            let mut tmp_acc_vec: Vec<_> = Vec::default();
            program!(@args_acc($builder, reg_man, tmp_acc_vec) $($tail)*);
            let tmp_cr = tmp_acc_vec.remove(0);

            let tmp_cr = $crate::macros::program::negate_bitmask($builder, tmp_cr, $control);
            let mut tmp_cb = $builder.condition_with(tmp_cr);

            // Now all the args are in acc_vec
            let mut tmp_results: Vec<_> = $func(&mut tmp_cb, tmp_acc_vec)?;

            let tmp_cr = tmp_cb.dissolve();
            let tmp_cr = $crate::macros::program::negate_bitmask($builder, tmp_cr, $control);

            tmp_results.push(tmp_cr);
            program!(@replace_registers($builder, reg_man, tmp_results));
            Ok(reg_man)
        });
        program!(@skip_to_next_program($builder, $program_acc) $($tail)*);
    };

    // Start parsing a program of the form "function(arg) [register <indices>, ...];"
    (@program($builder:expr, $program_acc:ident) $func:ident($funcargs:expr) $($tail:tt)*) => {
        let $program_acc = $program_acc.and_then(|mut reg_man| -> Result<$crate::macros::program::RegisterManager<_>, $crate::errors::CircuitError> {
            // Get all args
            let mut acc_vec: Vec<_> = Vec::default();
            program!(@args_acc($builder, reg_man, acc_vec) $($tail)*);

            // Now all the args are in acc_vec
            let tmp_results: Vec<_> = $func($builder, acc_vec, $funcargs)?;
            program!(@replace_registers($builder, reg_man, tmp_results));
            Ok(reg_man)
        });
        program!(@skip_to_next_program($builder, $program_acc) $($tail)*);
    };

    // Start parsing a program of the form "function [register <indices>, ...];"
    (@program($builder:expr, $program_acc:ident) $func:ident $($tail:tt)*) => {
        let $program_acc = $program_acc.and_then(|mut reg_man| -> Result<$crate::macros::program::RegisterManager<_>, $crate::errors::CircuitError> {
            // Get all args
            let mut acc_vec: Vec<_> = Vec::default();
            program!(@args_acc($builder, reg_man, acc_vec) $($tail)*);

            // Now all the args are in acc_vec
            let tmp_results: Vec<_> = $func($builder, acc_vec)?;
            program!(@replace_registers($builder, reg_man, tmp_results));
            Ok(reg_man)
        });
        program!(@skip_to_next_program($builder, $program_acc) $($tail)*);
    };

    // Skip past the register list and start running programs.
    (@skip_to_program($builder:expr, $reg_man:ident) $name:ident; $($tail:tt)*) => {
        program!(@program($builder, $reg_man) $($tail)*)
    };
    (@skip_to_program($builder:expr, $reg_man:ident) $name:ident, $($tail:tt)*) => {
        program!(@skip_to_program($builder, $reg_man) $($tail)*)
    };

    // (builder, register_1, ...; programs; ...)
    ($builder:expr, $($tail:tt)*) => {
        {
            // First reassign each name to a vec index
            let mut register_manager = $crate::macros::program::RegisterManager::new();
            program!(@splitter($builder, register_manager) $($tail)*);

            let program_acc = Ok(register_manager);
            program!(@skip_to_program($builder, program_acc) $($tail)*);

            program_acc.and_then(|mut register_manager| -> Result<_, $crate::errors::CircuitError> {
                program!(@joiner($builder, register_manager) $($tail)*);
                Ok(program!(@name_tuple () <- $($tail)*))
            })
        }
    };
}

/// Negate all the qubits in a register where the mask bit == 0.
pub fn negate_bitmask<P: Precision, CB: CliffordTBuilder<P>>(
    b: &mut CB,
    r: CB::Register,
    mask: u64,
) -> CB::Register {
    let rs = b.split_all_register(r);
    let (rs, _) = rs.into_iter().fold(
        (Vec::default(), mask),
        |(mut qubit_acc, mask_acc), qubit| {
            let lowest = mask_acc & 1;
            let qubit = if lowest == 0 { b.not(qubit) } else { qubit };
            qubit_acc.push(qubit);
            (qubit_acc, mask_acc >> 1)
        },
    );
    b.merge_registers(rs).unwrap()
}

/// A class which handles registers for the program macro in order to reduce code duplication.
#[derive(Default, Debug)]
pub struct RegisterManager<R: QubitRegister> {
    registers: Vec<(String, Vec<Option<R>>)>,
    reverse_lookup: Vec<Option<(usize, usize)>>,
}

impl<R: QubitRegister> RegisterManager<R> {
    pub fn new() -> Self {
        Self {
            registers: Vec::default(),
            reverse_lookup: Vec::default(),
        }
    }

    /// Add a register to the manager
    pub fn add_register<CB: CircuitBuilder<Register = R>>(
        &mut self,
        b: &mut CB,
        r: R,
        name: &str,
    ) -> RegisterDataWrapper {
        let register_wrapper = RegisterDataWrapper::new(&r, self.registers.len());
        let rs = b.split_all_register(r);
        let reverse_map: Vec<usize> = rs.iter().map(|r| r.indices()[0]).collect();
        let max_reverse = *reverse_map.iter().max().unwrap();
        let reg_vec = rs.into_iter().map(Some).collect();
        if max_reverse >= self.reverse_lookup.len() {
            self.reverse_lookup.resize(max_reverse + 1, None);
        }
        let reg_len = self.registers.len();
        reverse_map
            .into_iter()
            .enumerate()
            .for_each(|(rel_indx, qubit_indx)| {
                self.reverse_lookup[qubit_indx] = Some((reg_len, rel_indx));
            });

        self.registers.push((name.to_string(), reg_vec));
        register_wrapper
    }

    /// Get all qubits from a register
    pub fn get_full_register<CB: CircuitBuilder<Register = R>>(
        &mut self,
        b: &mut CB,
        rid: usize,
    ) -> CircuitResult<R> {
        let (name, registers_for_name) = &mut self.registers[rid];
        let rs = registers_for_name.iter_mut().enumerate().try_fold(
            Vec::default(),
            |mut acc, (index, op_r)| {
                let r = op_r.take().ok_or_else(|| {
                    CircuitError::new(format!(
                        "Failed to fetch {}[{}]. May have already been used on this line.",
                        name.clone(),
                        index
                    ))
                })?;
                acc.push(r);
                Ok(acc)
            },
        )?;
        b.merge_registers(rs)
            .ok_or_else(|| CircuitError::new("No register found"))
    }

    /// Get qubits with specific relative indices for a register.
    pub fn get_registers<'a, T, CB: CircuitBuilder<Register = R>>(
        &mut self,
        b: &mut CB,
        rid: usize,
        relative_indices: &'a [T],
    ) -> CircuitResult<R>
    where
        &'a T: Into<QubitIndices>,
    {
        let (name, registers_for_name) = &mut self.registers[rid];
        let rs = relative_indices
            .iter()
            .try_fold(Vec::default(), |mut acc, indices| {
                let indices: QubitIndices = indices.into();
                let rs = indices.get_indices().into_iter().try_fold(
                    Vec::default(),
                    |mut acc, index| {
                        let op_r = registers_for_name[index].take();
                        let r = op_r.ok_or_else(|| {
                            CircuitError::new(format!(
                                "Failed to fetch {}[{}]. May have already been used on this line.",
                                name.clone(),
                                index
                            ))
                        })?;
                        acc.push(r);
                        Ok(acc)
                    },
                )?;
                acc.push(rs);
                Ok(acc)
            })?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        b.merge_registers(rs)
            .ok_or_else(|| CircuitError::new("Register not found"))
    }

    /// Pop off a register
    pub fn pop_register<CB: CircuitBuilder<Register = R>>(&mut self, b: &mut CB) -> R {
        let (_, registers) = self.registers.pop().unwrap();
        let rs = registers
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        rs.iter().map(|r| r.indices()[0]).for_each(|indx| {
            self.reverse_lookup[indx] = None;
        });
        b.merge_registers(rs).unwrap()
    }

    /// Give qubits back to any registers.
    pub fn return_registers<CB: CircuitBuilder<Register = R>>(&mut self, b: &mut CB, rs: Vec<R>) {
        rs.into_iter()
            .flat_map(|r| b.split_all_register(r))
            .for_each(|r: R| {
                let (reg_indx, rel_indx) = self.reverse_lookup[r.indices()[0]].unwrap();
                let (_, qubits) = &mut self.registers[reg_indx];
                qubits[rel_indx] = Some(r);
            });
    }
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
    pub indices: Vec<usize>,
    /// Number of qubits in the register
    pub n: usize,
    /// Index of the register as arg to program!
    pub index: usize,
}

impl RegisterDataWrapper {
    /// Make a new RegisterDataWrapper for a register (with given index)
    pub fn new<R: QubitRegister>(r: &R, index: usize) -> Self {
        Self {
            indices: r.indices().to_vec(),
            n: r.n(),
            index,
        }
    }

    /// Number of qubits in the register
    pub fn n(&self) -> usize {
        self.n
    }
}
