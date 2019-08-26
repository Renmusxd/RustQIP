use crate::pipeline::{get_owned_opfns, StateModifierType};
use crate::state_ops::UnitaryOp;
use crate::unitary_decomposition::utils::transpose_sparse;
use crate::utils::get_flat_index;
use crate::{CircuitError, Complex, OpBuilder, Register, UnitaryBuilder};

/// Wrap a function to create a version compatible with `program!` as well as an inverse which is
/// also compatible.
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
/// wrap_and_invert!(wrap_cy, inv_cy, UnitaryBuilder::cy, ra, rb);
///
/// let (ra, rb) = program!(&mut b, ra, rb;
///     wrap_cy ra, rb[2];
///     inv_cy ra, rb[2];
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! wrap_and_invert {
    (pub $newfunc:ident, pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc, $($tail)*);
        invert_fn!(pub $newinvert, $newfunc);
    };
    ($newfunc:ident, pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc, $($tail)*);
        invert_fn!(pub $newinvert, $newfunc);
    };
    (pub $newfunc:ident, $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc, $($tail)*);
        invert_fn!($newinvert, $newfunc);
    };
    ($newfunc:ident, $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc, $($tail)*);
        invert_fn!($newinvert, $newfunc);
    }
}

/// Wrap a function to create a version compatible with `program!` as well as an inverse which is
/// also compatible.
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
/// wrap_fn!(wrap_cy, UnitaryBuilder::cy, ra, rb);
/// invert_fn!(inv_cy, wrap_cy);
///
/// let (ra, rb) = program!(&mut b, ra, rb;
///     wrap_cy ra, rb[2];
///     inv_cy ra, rb[2];
/// )?;
/// let r = b.merge(vec![ra, rb])?;
///
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! invert_fn {
    (pub $newinvert:ident, $func:expr) => {
        pub fn $newinvert(
            b: &mut dyn UnitaryBuilder,
            rs: Vec<Register>,
        ) -> Result<Vec<Register>, CircuitError> {
            $crate::inverter(b, rs, $func)
        }
    };
    ($newinvert:ident, $func:expr) => {
        fn $newinvert(
            b: &mut dyn UnitaryBuilder,
            rs: Vec<Register>,
        ) -> Result<Vec<Register>, CircuitError> {
            $crate::inverter(b, rs, $func)
        }
    };
}

/// Invert a circuit applied via the function f.
pub fn inverter<
    F: Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>,
>(
    b: &mut dyn UnitaryBuilder,
    rs: Vec<Register>,
    f: F,
) -> Result<Vec<Register>, CircuitError> {
    let original_indices: Vec<_> = rs.iter().map(|r| r.indices.clone()).collect();
    let mut inv_builder = OpBuilder::new();
    let new_rs: Vec<_> = original_indices
        .iter()
        .map(|indices| {
            let n = indices.len() as u64;
            inv_builder.register(n).unwrap()
        })
        .collect();
    let flat_indices: Vec<_> = original_indices.iter().flatten().cloned().collect();
    let new_rs = f(&mut inv_builder, new_rs)?;
    let end_reg = inv_builder.merge(new_rs)?;

    let reg = b.merge(rs)?;
    let reg = get_owned_opfns(end_reg)
        .into_iter()
        .map(|modifier| {
            let name = format!("Invert({})", modifier.name);
            match modifier.modifier {
                StateModifierType::UnitaryOp(op) => {
                    let inv_op = remap_indices(invert_op(op), &flat_indices);
                    (name, inv_op)
                }
                StateModifierType::SideChannelModifiers(_, _) => unimplemented!(),
                StateModifierType::MeasureState(_, _, _) => unimplemented!(),
                StateModifierType::StochasticMeasureState(_, _, _) => unimplemented!(),
            }
        })
        .rev()
        .try_fold(reg, |reg, (name, op)| {
            let indices = match &op {
                UnitaryOp::Matrix(indices, _) => indices.clone(),
                UnitaryOp::SparseMatrix(indices, _) => indices.clone(),
                UnitaryOp::Swap(a_indices, b_indices) => {
                    let vecs = [a_indices.clone(), b_indices.clone()];
                    vecs.iter().flatten().cloned().collect()
                }
                UnitaryOp::Control(c_indices, op_indices, _) => {
                    let vecs = [c_indices.clone(), op_indices.clone()];
                    vecs.iter().flatten().cloned().collect()
                }
                UnitaryOp::Function(x_indices, y_indices, _) => {
                    let vecs = [x_indices.clone(), y_indices.clone()];
                    vecs.iter().flatten().cloned().collect()
                }
            };
            if indices.len() == reg.n() as usize {
                b.merge_with_op(vec![reg], Some((name, op)))
            } else {
                let (sel_reg, reg) = b.split_absolute(reg, &indices)?;
                let sel_reg = b.merge_with_op(vec![sel_reg], Some((name, op)))?;
                b.merge(vec![sel_reg, reg])
            }
        })?;
    let (rs, _) = b.split_absolute_many(reg, &original_indices)?;
    Ok(rs)
}

fn invert_op(op: UnitaryOp) -> UnitaryOp {
    match op {
        UnitaryOp::Matrix(indices, mut mat) => {
            let n = indices.len() as u64;
            (0..1 << n).for_each(|row| {
                (0..row).for_each(|col| {
                    mat.swap(
                        get_flat_index(n, row, col) as usize,
                        get_flat_index(n, col, row) as usize,
                    );
                })
            });
            let mat = mat.into_iter().map(|v| v.conj()).collect();
            UnitaryOp::Matrix(indices, mat)
        }
        UnitaryOp::SparseMatrix(indices, mat) => {
            UnitaryOp::SparseMatrix(indices, transpose_sparse(mat))
        }
        UnitaryOp::Swap(a_indices, b_indices) => UnitaryOp::Swap(a_indices, b_indices),
        UnitaryOp::Control(c_indices, op_indices, op) => {
            UnitaryOp::Control(c_indices, op_indices, Box::new(invert_op(*op)))
        }
        UnitaryOp::Function(x_indices, y_indices, f) => {
            let n = (x_indices.len() + y_indices.len()) as u64;
            let mat = (0..1 << n)
                .map(|col| {
                    let (row, phase) = f(col);
                    let val = Complex { re: 0.0, im: phase }.exp();
                    vec![(row, val)]
                })
                .collect();
            let indices = x_indices.into_iter().chain(y_indices.into_iter()).collect();
            invert_op(UnitaryOp::SparseMatrix(indices, mat))
        }
    }
}

fn remap_indices(op: UnitaryOp, new_indices: &[u64]) -> UnitaryOp {
    let remap = |indices: Vec<u64>| -> Vec<u64> {
        indices
            .into_iter()
            .map(|indx| new_indices[indx as usize])
            .collect()
    };
    match op {
        UnitaryOp::Matrix(indices, mat) => UnitaryOp::Matrix(remap(indices), mat),
        UnitaryOp::SparseMatrix(indices, mat) => UnitaryOp::SparseMatrix(remap(indices), mat),
        UnitaryOp::Swap(a_indices, b_indices) => {
            UnitaryOp::Swap(remap(a_indices), remap(b_indices))
        }
        UnitaryOp::Control(c_indices, op_indices, op) => {
            let op = Box::new(remap_indices(*op, new_indices));
            UnitaryOp::Control(remap(c_indices), remap(op_indices), op)
        }
        UnitaryOp::Function(x_indices, y_indices, f) => {
            UnitaryOp::Function(remap(x_indices), remap(y_indices), f)
        }
    }
}

#[cfg(test)]
mod inverter_test {
    use super::*;
    use crate::boolean_circuits::arithmetic::{add, add_op};
    use crate::pipeline::InitialState;
    use crate::{run_debug, run_local_with_init, Precision, QuantumState};
    use num::One;

    fn test_inversion<
        F: Fn(&mut dyn UnitaryBuilder, Vec<Register>) -> Result<Vec<Register>, CircuitError>,
    >(
        b: &mut dyn UnitaryBuilder,
        rs: Vec<Register>,
        f: F,
    ) -> Result<(), CircuitError> {
        let n: u64 = rs.iter().map(|r| r.n()).sum();
        let indices: Vec<_> = (0..n).collect();
        let rs = f(b, rs)?;
        let rs = inverter(b, rs, f)?;
        let r = b.merge(rs)?;

        run_debug(&r)?;

        (0..1 << n).for_each(|indx| {
            let (state, _) =
                run_local_with_init::<f64>(&r, &[(indices.clone(), InitialState::Index(indx))])
                    .unwrap();
            let pos = state
                .get_state(false)
                .into_iter()
                .position(|v| v == Complex::one())
                .map(|pos| pos as u64);
            assert!(pos.is_some());
            assert_eq!(pos.unwrap(), indx);
        });
        Ok(())
    }

    #[test]
    fn test_invert_x() -> Result<(), CircuitError> {
        wrap_fn!(x_op, UnitaryBuilder::x, r);
        let mut b = OpBuilder::new();
        let r = b.qubit();
        test_inversion(&mut b, vec![r], x_op)
    }

    #[test]
    fn test_invert_y() -> Result<(), CircuitError> {
        wrap_fn!(x_op, UnitaryBuilder::y, r);
        let mut b = OpBuilder::new();
        let r = b.qubit();
        test_inversion(&mut b, vec![r], x_op)
    }

    #[test]
    fn test_invert_multi() -> Result<(), CircuitError> {
        fn gamma(
            b: &mut dyn UnitaryBuilder,
            ra: Register,
            rb: Register,
        ) -> Result<(Register, Register), CircuitError> {
            let (ra, rb) = b.cy(ra, rb);
            b.swap(ra, rb)
        }
        wrap_fn!(wrap_gamma, (gamma), ra, rb);
        let mut b = OpBuilder::new();
        let ra = b.qubit();
        let rb = b.qubit();
        test_inversion(&mut b, vec![ra, rb], wrap_gamma)
    }

    #[test]
    fn test_invert_add() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.qubit();
        let ra = b.qubit();
        let rb = b.register(2)?;
        test_inversion(&mut b, vec![rc, ra, rb], add_op)
    }

    #[test]
    fn test_invert_add_larger() -> Result<(), CircuitError> {
        let mut b = OpBuilder::new();
        let rc = b.register(2)?;
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        test_inversion(&mut b, vec![rc, ra, rb], add_op)
    }

    #[test]
    fn test_invert_and_wrap_add() -> Result<(), CircuitError> {
        wrap_and_invert!(add_op, inv_add, (add), rc, ra, rb);
        let mut b = OpBuilder::new();
        let rc = b.register(2)?;
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        let rs = add_op(&mut b, vec![rc, ra, rb])?;
        let rs = inv_add(&mut b, rs)?;
        let r = b.merge(rs)?;
        Ok(())
    }

    #[test]
    fn test_invert_add_larger_macro() -> Result<(), CircuitError> {
        invert_fn!(inv_add, add_op);
        let mut b = OpBuilder::new();
        let rc = b.register(2)?;
        let ra = b.register(2)?;
        let rb = b.register(3)?;
        let rs = inv_add(&mut b, vec![rc, ra, rb])?;
        let r = b.merge(rs)?;
        Ok(())
    }
}
