use crate::errors::CircuitError;
/// This module is meant to provide some cleaner apis for common circuit designs by allowing the
/// chaining of builder operations.
///
/// # Example
/// ```
/// use qip::*;
/// let mut b = OpBuilder::default();
/// let r = b.register(1)?;  // Make the Register, apply x, y, z and release.
/// let r = chain(&mut b, r).x().y().z().r();
/// ```
use crate::{Register, UnitaryBuilder};
use num::Complex;
use std::ops::Not;

/// Produce a chaining struct for a given Register `q`, operations will be applied using `b`.
pub fn chain<B: UnitaryBuilder>(b: &mut B, q: Register) -> SingleRegisterChain<B> {
    SingleRegisterChain::new(b, q)
}

/// Produce a chaining struct for a Register tuple `(qa, qb)`, operations will be applied using `b`.
pub fn chain_tuple<B: UnitaryBuilder>(b: &mut B, qa: Register, qb: Register) -> DoubleRegisterChain<B> {
    DoubleRegisterChain::new(b, qa, qb)
}

/// Produce a chaining struct for a Register vec `qs`, operations will be applied using `b`.
pub fn chain_vec<B: UnitaryBuilder>(b: &mut B, qs: Vec<Register>) -> VecRegisterChain<B> {
    VecRegisterChain::new(b, qs)
}

/// Chaining struct for a single Register (which may have multiple indices)
#[derive(Debug)]
pub struct SingleRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    q: Register,
}

/// Chaining struct for a pair of Registers (each may have multiple indices)
#[derive(Debug)]
pub struct DoubleRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    qa: Register,
    qb: Register,
}

/// Chaining struct for a vector of Registers (each may have multiple indices)
#[derive(Debug)]
pub struct VecRegisterChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    qs: Vec<Register>,
}

impl<'a, B: UnitaryBuilder> SingleRegisterChain<'a, B> {
    /// Make a new SingleQubitChain. Prefer to use `chain`.
    pub fn new(builder: &'a mut B, q: Register) -> Self {
        SingleRegisterChain::<'a, B> { builder, q }
    }
    /// Release the contained Register.
    pub fn release(self) -> Register {
        self.q()
    }
    /// Release the contained Register.
    pub fn q(self) -> Register {
        self.q
    }

    /// Split the Register, select the given indices and transfer them to a new Register, leave the
    /// remaining indices in another Register. This uses the relative indices (0 refers to whatever the
    /// first index of the contained qubit is).
    pub fn split(self, indices: Vec<u64>) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (qa, qb) = self.builder.split(self.q, indices)?;
        Ok(DoubleRegisterChain::new(self.builder, qa, qb))
    }
    /// Split the Register, select the given indices and transfer them to a new Register, leave the
    /// remaining indices in another Register. This uses the absolute indices (0 refers to the 0th
    /// absolute index, even if it isn't in the contained qubit, throwing an error).
    pub fn split_absolute(
        self,
        selected_indices: Vec<u64>,
    ) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (qa, qb) = self.builder.split_absolute(self.q, selected_indices)?;
        Ok(DoubleRegisterChain::new(self.builder, qa, qb))
    }
    /// Split each contained index into its own Register object, returns a chaining struct for the vec
    /// of resulting Registers.
    pub fn split_all(self) -> Result<VecRegisterChain<'a, B>, CircuitError> {
        let qs = self.builder.split_all(self.q);
        Ok(VecRegisterChain::new(self.builder, qs))
    }
    /// Apply a matrix operation to the contained Register.
    pub fn apply_mat(self, name: &str, mat: Vec<Complex<f64>>) -> Result<Self, CircuitError> {
        let q = self.builder.mat(name, self.q, mat)?;
        Ok(Self::new(self.builder, q))
    }
    /// Apply a sparse matrix operation to the contained Register.
    pub fn apply_sparse_mat(
        self,
        name: &str,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Self, CircuitError> {
        let q = self.builder.sparse_mat(name, self.q, mat, natural_order)?;
        Ok(Self::new(self.builder, q))
    }

    /// Apply the X operation to the contained Register.
    pub fn x(self) -> Self {
        let q = self.builder.x(self.q);
        Self::new(self.builder, q)
    }
    /// Apply the Y operation to the contained Register.
    pub fn y(self) -> Self {
        let q = self.builder.y(self.q);
        Self::new(self.builder, q)
    }
    /// Apply the Z operation to the contained Register.
    pub fn z(self) -> Self {
        let q = self.builder.z(self.q);
        Self::new(self.builder, q)
    }
    /// Apply a H to the contained Register.
    pub fn hadamard(self) -> Self {
        let q = self.builder.hadamard(self.q);
        Self::new(self.builder, q)
    }
    /// Map the Register by the given function.
    pub fn apply(self, f: impl FnOnce(&mut B, Register) -> Register) -> Self {
        let q = f(self.builder, self.q);
        Self::new(self.builder, q)
    }
    /// Map the Register by the given function, resulting in two Registers.
    pub fn apply_cut(
        self,
        f: impl FnOnce(&mut B, Register) -> (Register, Register),
    ) -> DoubleRegisterChain<'a, B> {
        let (qa, qb) = f(self.builder, self.q);
        DoubleRegisterChain::new(self.builder, qa, qb)
    }
    /// Map the Register by the given function, resulting in a vec of Registers.
    pub fn apply_split(self, f: impl FnOnce(&mut B, Register) -> Vec<Register>) -> VecRegisterChain<'a, B> {
        let qs = f(self.builder, self.q);
        VecRegisterChain::new(self.builder, qs)
    }
}

impl<'a, B: UnitaryBuilder> Not for SingleRegisterChain<'a, B> {
    type Output = SingleRegisterChain<'a, B>;

    fn not(self) -> Self {
        let q = self.builder.not(self.q);
        Self::new(self.builder, q)
    }
}

impl<'a, B: UnitaryBuilder> DoubleRegisterChain<'a, B> {
    /// Make a new `DoubleRegisterChain`, prefer to use `chain_tuple`.
    pub fn new(builder: &'a mut B, qa: Register, qb: Register) -> Self {
        DoubleRegisterChain::<'a, B> { builder, qa, qb }
    }
    /// Release the contained Register tuple
    pub fn release(self) -> (Register, Register) {
        self.qab()
    }
    /// Release the contained Register tuple
    pub fn qab(self) -> (Register, Register) {
        (self.qa, self.qb)
    }
    /// Merge the contained Register tuple into a single Register, wrap in a chaining struct.
    pub fn merge(self) -> SingleRegisterChain<'a, B> {
        let q = self.builder.merge(vec![self.qa, self.qb]);
        SingleRegisterChain::new(self.builder, q)
    }
    /// Split all the indices for each Register into their own Registers, returned the chained struct for
    /// the vec of Registers.
    pub fn split_all(self) -> VecRegisterChain<'a, B> {
        let q = self.builder.merge(vec![self.qa, self.qb]);
        let qs = self.builder.split_all(q);
        VecRegisterChain::new(self.builder, qs)
    }
    /// Apply a swap op to the contained Registers, will only succeed of the Registers are of equal size.
    pub fn swap(self) -> Result<Self, CircuitError> {
        let (qa, qb) = self.builder.swap(self.qa, self.qb)?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Swap the positions of the contained Registers. This is not a quantum operation, rather a
    /// a bookkeeping one.
    pub fn physical_swap(self) -> Self {
        Self::new(self.builder, self.qb, self.qa)
    }
    /// Apply a function operation to the contained Registers, the first will act as the readin
    /// register and the second as the output.
    pub fn apply_function_op(
        self,
        f: impl Fn(u64) -> (u64, f64) + Send + Sync + 'static,
    ) -> Result<Self, CircuitError> {
        let (qa, qb) = self.builder.apply_function(self.qa, self.qb, Box::new(f))?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Apply a function op which has been already boxed.
    pub fn apply_boxed_function_op(
        self,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<Self, CircuitError> {
        let (qa, qb) = self.builder.apply_function(self.qa, self.qb, f)?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Apply a function which outputs a single Register.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Register, Register) -> Register,
    ) -> SingleRegisterChain<'a, B> {
        let q = f(self.builder, self.qa, self.qb);
        SingleRegisterChain::new(self.builder, q)
    }
    /// Apply a function which outputs a tuple of Registers.
    pub fn apply(self, f: impl FnOnce(&mut B, Register, Register) -> (Register, Register)) -> Self {
        let (qa, qb) = f(self.builder, self.qa, self.qb);
        Self::new(self.builder, qa, qb)
    }
    /// Apply a function which outputs a vector of Registers.
    pub fn apply_split(
        self,
        f: impl FnOnce(&mut B, Register, Register) -> Vec<Register>,
    ) -> VecRegisterChain<'a, B> {
        let qs = f(self.builder, self.qa, self.qb);
        VecRegisterChain::new(self.builder, qs)
    }
}

impl<'a, B: UnitaryBuilder> VecRegisterChain<'a, B> {
    /// Make a new `VecRegisterChain`, prefer to use `chain_vec`.
    pub fn new(builder: &'a mut B, qs: Vec<Register>) -> Self {
        VecRegisterChain::<'a, B> { builder, qs }
    }
    /// Release the contained vec of Registers.
    pub fn release(self) -> Vec<Register> {
        self.qs()
    }
    /// Release the contained vec of Registers.
    pub fn qs(self) -> Vec<Register> {
        self.qs
    }
    /// Merge the contained vec of Registers into a single Register.
    pub fn merge(self) -> SingleRegisterChain<'a, B> {
        let q = self.builder.merge(self.qs);
        SingleRegisterChain::new(self.builder, q)
    }
    /// Partition the contained Registers into two groups by their index in the underlying vector.
    /// Merge each group into a Register and produce a chained struct for the tuple.
    pub fn partition_by_relative(
        self,
        f: impl Fn(u64) -> bool,
    ) -> Result<DoubleRegisterChain<'a, B>, CircuitError> {
        let (a, b): (Vec<_>, Vec<_>) = self
            .qs
            .into_iter()
            .enumerate()
            .partition(|(i, _)| f(*i as u64));

        if a.is_empty() {
            CircuitError::make_str_err("Partition must provide at least one Register to first entry.")
        } else if b.is_empty() {
            CircuitError::make_str_err("Partition must provide at least one Register to second entry.")
        } else {
            let f = |vs: Vec<(usize, Register)>| -> Vec<Register> {
                vs.into_iter().map(|(_, q)| q).collect()
            };
            let qa = self.builder.merge(f(a));
            let qb = self.builder.merge(f(b));
            Ok(DoubleRegisterChain::new(self.builder, qa, qb))
        }
    }
    /// Flatten the Registers: make a chain struct representing the vec of all single-index
    /// Registers which can be made from the current set of owned indices.
    /// Acts as flatten would on a vec of vec of indices.
    pub fn flatten(self) -> Self {
        let qs = self.qs;
        let builder = self.builder;
        let qs: Vec<_> = qs
            .into_iter()
            .map(|q| builder.split_all(q))
            .flatten()
            .collect();
        Self::new(builder, qs)
    }
    /// Apply a function which outputs a single Register.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Vec<Register>) -> Register,
    ) -> SingleRegisterChain<'a, B> {
        let q = f(self.builder, self.qs);
        SingleRegisterChain::new(self.builder, q)
    }
    /// Apply a function which outputs a tuple of Registers.
    pub fn apply_partition(
        self,
        f: impl FnOnce(&mut B, Vec<Register>) -> (Register, Register),
    ) -> DoubleRegisterChain<'a, B> {
        let (qa, qb) = f(self.builder, self.qs);
        DoubleRegisterChain::new(self.builder, qa, qb)
    }
    /// Apply a function which outputs a vector of Registers.
    pub fn apply(self, f: impl FnOnce(&mut B, Vec<Register>) -> Vec<Register>) -> Self {
        let qs = f(self.builder, self.qs);
        Self::new(self.builder, qs)
    }
}
