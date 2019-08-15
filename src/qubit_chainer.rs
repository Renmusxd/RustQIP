use crate::errors::CircuitError;
/// This module is meant to provide some cleaner apis for common circuit designs by allowing the
/// chaining of builder operations.
///
/// # Example
/// ```
/// use qip::*;
/// let mut b = OpBuilder::default();
/// let r = b.r(1);  // Make the Register, apply x, y, z and release.
/// let r = chain(&mut b, r).x().y().z().r();
/// ```
use crate::{Register, UnitaryBuilder};
use num::Complex;
use std::ops::Not;

/// Produce a chaining struct for a given Register `r`, operations will be applied using `b`.
pub fn chain<B: UnitaryBuilder>(b: &mut B, r: Register) -> SingleRegisterChain<B> {
    SingleRegisterChain::new(b, r)
}

/// Produce a chaining struct for a Register tuple `(ra, rb)`, operations will be applied using `b`.
pub fn chain_tuple<B: UnitaryBuilder>(
    b: &mut B,
    ra: Register,
    rb: Register,
) -> DoubleRegisterChain<B> {
    DoubleRegisterChain::new(b, ra, rb)
}

/// Produce a chaining struct for a Register vec `rs`, operations will be applied using `b`.
pub fn chain_vec<B: UnitaryBuilder>(b: &mut B, rs: Vec<Register>) -> VecRegisterChain<B> {
    VecRegisterChain::new(b, rs)
}

/// Chaining struct for a single Register (which may have multiple indices)
#[derive(Debug)]
pub struct SingleRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    r: Register,
}

/// Chaining struct for a pair of Registers (each may have multiple indices)
#[derive(Debug)]
pub struct DoubleRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    ra: Register,
    rb: Register,
}

/// Chaining struct for a vector of Registers (each may have multiple indices)
#[derive(Debug)]
pub struct VecRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    rs: Vec<Register>,
}

impl<'a, B: UnitaryBuilder> SingleRegisterChain<'a, B> {
    /// Make a new SingleRegisterChain. Prefer to use `chain`.
    pub fn new(builder: &'a mut B, r: Register) -> Self {
        SingleRegisterChain::<'a, B> { builder, r }
    }
    /// Release the contained Register.
    pub fn release(self) -> Register {
        self.r()
    }
    /// Release the contained Register.
    pub fn r(self) -> Register {
        self.r
    }

    /// Split the Register, select the given indices and transfer them to a new Register, leave the
    /// remaining indices in another Register. This uses the relative indices (0 refers to whatever the
    /// first index of the contained Register is).
    pub fn split(self, indices: Vec<u64>) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (ra, rb) = self.builder.split(self.r, indices)?;
        Ok(DoubleRegisterChain::new(self.builder, ra, rb))
    }
    /// Split the Register, select the given indices and transfer them to a new Register, leave the
    /// remaining indices in another Register. This uses the absolute indices (0 refers to the 0th
    /// absolute index, even if it isn't in the contained Register, throwing an error).
    pub fn split_absolute(
        self,
        selected_indices: Vec<u64>,
    ) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (ra, rb) = self.builder.split_absolute(self.r, selected_indices)?;
        Ok(DoubleRegisterChain::new(self.builder, ra, rb))
    }
    /// Split each contained index into its own Register object, returns a chaining struct for the vec
    /// of resulting Registers.
    pub fn split_all(self) -> Result<VecRegisterChain<'a, B>, CircuitError> {
        let rs = self.builder.split_all(self.r);
        Ok(VecRegisterChain::new(self.builder, rs))
    }
    /// Apply a matrix operation to the contained Register.
    pub fn apply_mat(self, name: &str, mat: Vec<Complex<f64>>) -> Result<Self, CircuitError> {
        let r = self.builder.mat(name, self.r, mat)?;
        Ok(Self::new(self.builder, r))
    }
    /// Apply a sparse matrix operation to the contained Register.
    pub fn apply_sparse_mat(
        self,
        name: &str,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Self, CircuitError> {
        let r = self.builder.sparse_mat(name, self.r, mat, natural_order)?;
        Ok(Self::new(self.builder, r))
    }

    /// Apply the X operation to the contained Register.
    pub fn x(self) -> Self {
        let r = self.builder.x(self.r);
        Self::new(self.builder, r)
    }
    /// Apply the Y operation to the contained Register.
    pub fn y(self) -> Self {
        let r = self.builder.y(self.r);
        Self::new(self.builder, r)
    }
    /// Apply the Z operation to the contained Register.
    pub fn z(self) -> Self {
        let r = self.builder.z(self.r);
        Self::new(self.builder, r)
    }
    /// Apply a H to the contained Register.
    pub fn hadamard(self) -> Self {
        let r = self.builder.hadamard(self.r);
        Self::new(self.builder, r)
    }
    /// Map the Register by the given function.
    pub fn apply(self, f: impl FnOnce(&mut B, Register) -> Register) -> Self {
        let r = f(self.builder, self.r);
        Self::new(self.builder, r)
    }
    /// Map the Register by the given function, resulting in two Registers.
    pub fn apply_cut(
        self,
        f: impl FnOnce(&mut B, Register) -> (Register, Register),
    ) -> DoubleRegisterChain<'a, B> {
        let (ra, rb) = f(self.builder, self.r);
        DoubleRegisterChain::new(self.builder, ra, rb)
    }
    /// Map the Register by the given function, resulting in a vec of Registers.
    pub fn apply_split(
        self,
        f: impl FnOnce(&mut B, Register) -> Vec<Register>,
    ) -> VecRegisterChain<'a, B> {
        let rs = f(self.builder, self.r);
        VecRegisterChain::new(self.builder, rs)
    }
}

impl<'a, B: UnitaryBuilder> Not for SingleRegisterChain<'a, B> {
    type Output = SingleRegisterChain<'a, B>;

    fn not(self) -> Self {
        let r = self.builder.not(self.r);
        Self::new(self.builder, r)
    }
}

impl<'a, B: UnitaryBuilder> DoubleRegisterChain<'a, B> {
    /// Make a new `DoubleRegisterChain`, prefer to use `chain_tuple`.
    pub fn new(builder: &'a mut B, ra: Register, rb: Register) -> Self {
        DoubleRegisterChain::<'a, B> { builder, ra, rb }
    }
    /// Release the contained Register tuple
    pub fn release(self) -> (Register, Register) {
        self.rab()
    }
    /// Release the contained Register tuple
    pub fn rab(self) -> (Register, Register) {
        (self.ra, self.rb)
    }
    /// Merge the contained Register tuple into a single Register, wrap in a chaining struct.
    pub fn merge(self) -> SingleRegisterChain<'a, B> {
        let r = self.builder.merge(vec![self.ra, self.rb]);
        SingleRegisterChain::new(self.builder, r)
    }
    /// Split all the indices for each Register into their own Registers, returned the chained struct for
    /// the vec of Registers.
    pub fn split_all(self) -> VecRegisterChain<'a, B> {
        let r = self.builder.merge(vec![self.ra, self.rb]);
        let qs = self.builder.split_all(r);
        VecRegisterChain::new(self.builder, qs)
    }
    /// Apply a swap op to the contained Registers, will only succeed of the Registers are of equal size.
    pub fn swap(self) -> Result<Self, CircuitError> {
        let (ra, rb) = self.builder.swap(self.ra, self.rb)?;
        Ok(Self::new(self.builder, ra, rb))
    }
    /// Swap the positions of the contained Registers. This is not a quantum operation, rather a
    /// a bookkeeping one.
    pub fn physical_swap(self) -> Self {
        Self::new(self.builder, self.rb, self.ra)
    }
    /// Apply a function operation to the contained Registers, the first will act as the readin
    /// register and the second as the output.
    pub fn apply_function_op(
        self,
        f: impl Fn(u64) -> (u64, f64) + Send + Sync + 'static,
    ) -> Result<Self, CircuitError> {
        let (ra, rb) = self.builder.apply_function(self.ra, self.rb, Box::new(f))?;
        Ok(Self::new(self.builder, ra, rb))
    }
    /// Apply a function op which has been already boxed.
    pub fn apply_boxed_function_op(
        self,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<Self, CircuitError> {
        let (ra, rb) = self.builder.apply_function(self.ra, self.rb, f)?;
        Ok(Self::new(self.builder, ra, rb))
    }
    /// Apply a function which outputs a single Register.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Register, Register) -> Register,
    ) -> SingleRegisterChain<'a, B> {
        let r = f(self.builder, self.ra, self.rb);
        SingleRegisterChain::new(self.builder, r)
    }
    /// Apply a function which outputs a tuple of Registers.
    pub fn apply(self, f: impl FnOnce(&mut B, Register, Register) -> (Register, Register)) -> Self {
        let (ra, rb) = f(self.builder, self.ra, self.rb);
        Self::new(self.builder, ra, rb)
    }
    /// Apply a function which outputs a vector of Registers.
    pub fn apply_split(
        self,
        f: impl FnOnce(&mut B, Register, Register) -> Vec<Register>,
    ) -> VecRegisterChain<'a, B> {
        let rs = f(self.builder, self.ra, self.rb);
        VecRegisterChain::new(self.builder, rs)
    }
}

impl<'a, B: UnitaryBuilder> VecRegisterChain<'a, B> {
    /// Make a new `VecRegisterChain`, prefer to use `chain_vec`.
    pub fn new(builder: &'a mut B, rs: Vec<Register>) -> Self {
        VecRegisterChain::<'a, B> { builder, rs }
    }
    /// Release the contained vec of Registers.
    pub fn release(self) -> Vec<Register> {
        self.rs()
    }
    /// Release the contained vec of Registers.
    pub fn rs(self) -> Vec<Register> {
        self.rs
    }
    /// Merge the contained vec of Registers into a single Register.
    pub fn merge(self) -> SingleRegisterChain<'a, B> {
        let r = self.builder.merge(self.rs);
        SingleRegisterChain::new(self.builder, r)
    }
    /// Partition the contained Registers into two groups by their index in the underlying vector.
    /// Merge each group into a Register and produce a chained struct for the tuple.
    pub fn partition_by_relative(
        self,
        f: impl Fn(u64) -> bool,
    ) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (a, b): (Vec<_>, Vec<_>) = self
            .rs
            .into_iter()
            .enumerate()
            .partition(|(i, _)| f(*i as u64));

        if a.is_empty() {
            CircuitError::make_str_err(
                "Partition must provide at least one Register to first entry.",
            )
        } else if b.is_empty() {
            CircuitError::make_str_err(
                "Partition must provide at least one Register to second entry.",
            )
        } else {
            let f = |vs: Vec<(usize, Register)>| -> Vec<Register> {
                vs.into_iter().map(|(_, r)| r).collect()
            };
            let ra = self.builder.merge(f(a));
            let rb = self.builder.merge(f(b));
            Ok(DoubleRegisterChain::new(self.builder, ra, rb))
        }
    }
    /// Flatten the Registers: make a chain struct representing the vec of all single-index
    /// Registers which can be made from the current set of owned indices.
    /// Acts as flatten would on a vec of vec of indices.
    pub fn flatten(self) -> Self {
        let rs = self.rs;
        let builder = self.builder;
        let qs: Vec<_> = rs
            .into_iter()
            .map(|r| builder.split_all(r))
            .flatten()
            .collect();
        Self::new(builder, qs)
    }
    /// Apply a function which outputs a single Register.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Vec<Register>) -> Register,
    ) -> SingleRegisterChain<'a, B> {
        let r = f(self.builder, self.rs);
        SingleRegisterChain::new(self.builder, r)
    }
    /// Apply a function which outputs a tuple of Registers.
    pub fn apply_partition(
        self,
        f: impl FnOnce(&mut B, Vec<Register>) -> (Register, Register),
    ) -> DoubleRegisterChain<'a, B> {
        let (ra, rb) = f(self.builder, self.rs);
        DoubleRegisterChain::new(self.builder, ra, rb)
    }
    /// Apply a function which outputs a vector of Registers.
    pub fn apply(self, f: impl FnOnce(&mut B, Vec<Register>) -> Vec<Register>) -> Self {
        let rs = f(self.builder, self.rs);
        Self::new(self.builder, rs)
    }
}
