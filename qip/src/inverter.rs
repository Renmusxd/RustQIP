use crate::builder_traits::{
    AdvancedCircuitBuilder, CircuitBuilder, QubitRegister, SplitManyResult, Subcircuitable,
};
use crate::conditioning::{Conditionable, ConditionableSubcircuit};
use crate::errors::CircuitResult;
use crate::types::Precision;

/// A trait which recursively requires that subcircuit builders also implement the traits the
/// original circuit builders implemented. This allows passing to functions which may arbitrarily
/// call other functions without type-tracking the depth of the stack.
pub trait RecursiveCircuitBuilder<P: Precision>:
    Invertable<SimilarBuilder = Self::RecursiveSimilarBuilder>
    + Conditionable
    + AdvancedCircuitBuilder<P>
    + Subcircuitable
    + ConditionableSubcircuit
{
    /// The similar builder which also implements the recursive circuit builder, it may be the same
    /// type or different.
    type RecursiveSimilarBuilder: RecursiveCircuitBuilder<P>
        + Subcircuitable<Subcircuit = Self::Subcircuit>;
}

/// An Invertable circuit builder must be able to produce a similar circuit builder with the
/// `new_simiar` call. This subcircuit builder can be used to make circuits that Self can invert
/// and then apply.
pub trait Invertable: Subcircuitable {
    /// A similar circuit builder which can be used to construct circuits for the parent.
    type SimilarBuilder: Subcircuitable<Subcircuit = Self::Subcircuit>;

    /// Make a similar circuit builder.
    fn new_similar(&self) -> Self::SimilarBuilder;
    /// Take the output of the similar circuit builder and invert it.
    fn invert_subcircuit(sc: Self::Subcircuit) -> CircuitResult<Self::Subcircuit>;
    /// Apply the inverted subcircuit to a register.
    fn apply_inverted_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        r: Self::Register,
    ) -> CircuitResult<Self::Register> {
        let sc = Self::invert_subcircuit(sc)?;
        self.apply_subcircuit(sc, r)
    }
}

/// Invert the circuit made by `f` using arguments `t`. Apply the inverted circuit to registers `rs`
/// using circuitbuilder `cb`.
pub fn inverter_args<T, CB, F>(
    cb: &mut CB,
    rs: Vec<CB::Register>,
    f: F,
    t: T,
) -> CircuitResult<Vec<CB::Register>>
where
    CB: Invertable,
    F: Fn(
        &mut CB::SimilarBuilder,
        Vec<<CB::SimilarBuilder as CircuitBuilder>::Register>,
        T,
    ) -> CircuitResult<Vec<<CB::SimilarBuilder as CircuitBuilder>::Register>>,
{
    let mut sub_cb = cb.new_similar();
    let sub_rs = rs
        .iter()
        .map(|r| sub_cb.register(r.n_nonzero()))
        .collect::<_>();
    let _ = f(&mut sub_cb, sub_rs, t)?;
    let subcircuit = sub_cb.make_subcircuit()?;
    let (_, ranges) = rs
        .iter()
        .map(|r| r.n())
        .fold((0, vec![]), |(n, mut acc), rn| {
            acc.push(n..n + rn);
            (n + rn, acc)
        });
    let r = cb.merge_registers(rs).unwrap();
    let r = cb.apply_inverted_subcircuit(subcircuit, r)?;
    match cb.split_relative_index_groups(r, ranges) {
        SplitManyResult::AllSelected(rs) => Ok(rs),
        SplitManyResult::Remaining(_, _) => unreachable!(),
    }
}

/// Invert the circuit made by `f`. Apply the inverted circuit to registers `rs` using
/// circuitbuilder `cb`.
pub fn inverter<CB, F>(cb: &mut CB, r: Vec<CB::Register>, f: F) -> CircuitResult<Vec<CB::Register>>
where
    CB: Invertable,
    F: Fn(
        &mut CB::SimilarBuilder,
        Vec<<CB::SimilarBuilder as CircuitBuilder>::Register>,
    ) -> CircuitResult<Vec<<CB::SimilarBuilder as CircuitBuilder>::Register>>,
{
    inverter_args(cb, r, |r, cb, _| f(r, cb), ())
}

#[cfg(test)]
mod inverter_test {
    // use super::*;
    // use crate::builder::Qudit;
    // use crate::macros::program_ops::*;
    // use crate::prelude::*;
    // use num_traits::One;
    //
    // fn test_inversion<F, P>(b: &mut LocalBuilder<P>, rs: Vec<Qudit>, f: F) -> CircuitResult<()>
    // where
    //     P: Precision,
    //     F: Fn(&mut LocalBuilder<P>, Vec<Qudit>) -> CircuitResult<Vec<Qudit>>,
    // {
    //     let rs = f(b, rs)?;
    //     let rs = inverter(b, rs, f)?;
    //     let r = b
    //         .merge_registers(rs)
    //         .ok_or(CircuitError::new("No registers returned."))?;
    //     let indices = (0..b.n()).collect::<Vec<_>>();
    //     (0..1 << b.n()).for_each(|indx| {
    //         let (state, _) = b.calculate_state_with_init([(&r, indx)]);
    //         let pos = state.into_iter().position(|v| v == Complex::one());
    //         // .map(|pos| flip_bits(n as usize, pos as u64));
    //         assert!(pos.is_some());
    //         assert_eq!(pos.unwrap(), indx);
    //     });
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_invert_x() -> Result<(), CircuitError> {
    //     wrap_fn!(x_op, x, r);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let r = b.qubit();
    //     test_inversion(&mut b, vec![r], x_op)
    // }
    //
    // #[test]
    // fn test_invert_y() -> Result<(), CircuitError> {
    //     wrap_fn!(y_op, y, r);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let r = b.qubit();
    //     test_inversion(&mut b, vec![r], y_op)
    // }
    //
    // #[test]
    // fn test_invert_multi() -> Result<(), CircuitError> {
    //     fn gamma(
    //         b: &mut dyn UnitaryBuilder,
    //         ra: Register,
    //         rb: Register,
    //     ) -> Result<(Register, Register), CircuitError> {
    //         let (ra, rb) = b.cy(ra, rb);
    //         b.swap(ra, rb)
    //     }
    //     wrap_fn!(wrap_gamma, (gamma), ra, rb);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let ra = b.qubit();
    //     let rb = b.qubit();
    //     test_inversion(&mut b, vec![ra, rb], wrap_gamma)
    // }
    //
    // #[test]
    // fn test_invert_add() -> Result<(), CircuitError> {
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.qubit();
    //     let ra = b.qubit();
    //     let rb = b.register(2)?;
    //     test_inversion(&mut b, vec![rc, ra, rb], add_op)
    // }
    //
    // #[test]
    // fn test_invert_add_larger() -> Result<(), CircuitError> {
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.register(2)?;
    //     let ra = b.register(2)?;
    //     let rb = b.register(3)?;
    //     test_inversion(&mut b, vec![rc, ra, rb], add_op)
    // }
    //
    // #[test]
    // fn test_invert_and_wrap_add() -> Result<(), CircuitError> {
    //     wrap_and_invert!(add_op, inv_add, (add), rc, ra, rb);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.register(2)?;
    //     let ra = b.register(2)?;
    //     let rb = b.register(3)?;
    //     let rs = add_op(&mut b, vec![rc, ra, rb])?;
    //     let rs = inv_add(&mut b, rs)?;
    //     let _r = b.merge(rs)?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_invert_add_larger_macro() -> Result<(), CircuitError> {
    //     invert_fn!(inv_add, add_op);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let rc = b.register(2)?;
    //     let ra = b.register(2)?;
    //     let rb = b.register(3)?;
    //     let rs = inv_add(&mut b, vec![rc, ra, rb])?;
    //     let _r = b.merge(rs)?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_invert_and_wrap_rz() -> Result<(), CircuitError> {
    //     wrap_and_invert!(rz_op(theta: f64), inv_rz, UnitaryBuilder::rz, r);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let r = b.qubit();
    //     let rs = rz_op(&mut b, vec![r], 1.0)?;
    //     let _rs = inv_rz(&mut b, rs, 1.0)?;
    //     Ok(())
    // }
    //
    // #[test]
    // fn test_invert_and_wrap_rz_generic() -> Result<(), CircuitError> {
    //     fn rz<T: Into<f64>>(b: &mut dyn UnitaryBuilder, r: Register, theta: T) -> Register {
    //         b.rz(r, theta.into())
    //     }
    //     wrap_and_invert!(rz_op[T: Into<f64>](theta: T), inv_rz, rz, r);
    //     let mut b = LocalBuilder::<f64>::default();
    //     let r = b.qubit();
    //     let rs = rz_op(&mut b, vec![r], 1.0)?;
    //     let _rs = inv_rz(&mut b, rs, 1.0)?;
    //     Ok(())
    // }
}
