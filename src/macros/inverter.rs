use crate::builder_traits::{CircuitBuilder, QubitRegister, SplitManyResult, Subcircuitable};
use crate::errors::CircuitResult;

pub trait Invertable: Subcircuitable {
    type SimilarBuilder: Subcircuitable<Subcircuit = Self::Subcircuit>;

    fn new_similar(&self) -> Self::SimilarBuilder;
    fn invert_subcircuit(sc: Self::Subcircuit) -> CircuitResult<Self::Subcircuit>;
    fn apply_inverted_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        r: Self::Register,
    ) -> CircuitResult<Self::Register> {
        let sc = Self::invert_subcircuit(sc)?;
        self.apply_subcircuit(sc, r)
    }
}

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
    let rns = rs.iter().map(|r| r.n()).collect::<Vec<_>>();
    let r = cb.merge_registers(rs).unwrap();
    let r = cb.apply_inverted_subcircuit(subcircuit, r)?;
    let (_, ranges) = rns.into_iter().fold((0, vec![]), |(n, mut acc), rn| {
        acc.push(n..n + rn);
        (n + rn, acc)
    });
    match cb.split_relative_index_groups(r, ranges) {
        SplitManyResult::AllSelected(rs) => Ok(rs),
        SplitManyResult::Remaining(_, _) => unreachable!(),
    }
}

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

/// Wrap a function to create a version compatible with `program!` as well as an inverse which is
/// also compatible.
#[macro_export]
macro_rules! invert_fn {
    (pub $newinvert:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $func:expr) => {
        /// Invert the given function.
        pub fn $newinvert<CB: $($typetail)*>(
            b: &mut CB,
            rs: Vec<CB::Register>,
            $arg: $argtype,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter_args(b, rs, $func, $arg)
        }
    };
    ($newinvert:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $func:expr) => {
        fn $newinvert<CB: $($typetail)*>(
            b: &mut CB,
            rs: Vec<CB::Register>,
            $arg: $argtype,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter_args(b, rs, $func, $arg)
        }
    };
    (pub $newinvert:ident($arg:ident: $argtype:ident), $func:expr) => {
        /// Invert the given function.
        pub fn $newinvert<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(
            b: &mut CB,
            rs: Vec<CB::Register>,
            $arg: $argtype,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter_args(b, rs, $func, $arg)
        }
    };
    ($newinvert:ident($arg:ident: $argtype:ident), $func:expr) => {
        fn $newinvert<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(
            b: &mut CB,
            rs: Vec<CB::Register>,
            $arg: $argtype,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter_args(b, rs, $func, $arg)
        }
    };
    (pub $newinvert:ident, $func:expr) => {
        /// Invert the given function.
        pub fn $newinvert<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(
            b: &mut CB,
            rs: Vec<CB::Register>,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter(b, rs, $func)
        }
    };
    ($newinvert:ident, $func:expr) => {
        fn $newinvert<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(
            b: &mut CB,
            rs: Vec<CB::Register>,
        ) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            $crate::macros::inverter::inverter(b, rs, $func)
        }
    };
}

/// Wrap a function to create a version compatible with `program!` as well as an inverse which is
/// also compatible.
#[macro_export]
macro_rules! wrap_and_invert {
    (pub $newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc[$($typetail)*]($arg: $argtype), $($tail)*);
        invert_fn!(pub $newinvert[$($typetail)*]($arg: $argtype), $newfunc);
    };
    ($newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc[$($typetail)*]($arg: $argtype), $($tail)*);
        invert_fn!(pub $newinvert[$($typetail)*]($arg: $argtype), $newfunc);
    };
    (pub $newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc[$($typetail)*]($arg: $argtype), $($tail)*);
        invert_fn!($newinvert[$($typetail)*]($arg: $argtype), $newfunc);
    };
    ($newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc[$($typetail)*]($arg: $argtype), $($tail)*);
        invert_fn!($newinvert[$($typetail)*]($arg: $argtype), $newfunc);
    };
    (pub $newfunc:ident($arg:ident: $argtype:ident), pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc($arg: $argtype), $($tail)*);
        invert_fn!(pub $newinvert($arg: $argtype), $newfunc);
    };
    ($newfunc:ident($arg:ident: $argtype:ident), pub $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc($arg: $argtype), $($tail)*);
        invert_fn!(pub $newinvert($arg: $argtype), $newfunc);
    };
    (pub $newfunc:ident($arg:ident: $argtype:ident), $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!(pub $newfunc($arg: $argtype), $($tail)*);
        invert_fn!($newinvert($arg: $argtype), $newfunc);
    };
    ($newfunc:ident($arg:ident: $argtype:ident), $newinvert:ident, $($tail:tt)*) => {
        wrap_fn!($newfunc($arg: $argtype), $($tail)*);
        invert_fn!($newinvert($arg: $argtype), $newfunc);
    };
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
