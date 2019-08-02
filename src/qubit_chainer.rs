/// This module is meant to provide some cleaner apis for common circuit designs by allowing the
/// chaining of builder operations.
///
/// # Example
/// ```
/// use qip::*;
/// let mut b = OpBuilder::default();
/// let q = b.q(1);  // Make the qubit, apply x, y, z and release.
/// let q = chain(&mut b, q).x().y().z().q();
/// ```
use crate::{Qubit, UnitaryBuilder};
use num::Complex;

/// Produce a chaining struct for a given qubit `q`, operations will be applied using `b`.
pub fn chain<B: UnitaryBuilder>(b: &mut B, q: Qubit) -> SingleQubitChain<B> {
    SingleQubitChain::new(b, q)
}

/// Produce a chaining struct for a qubit tuple `(qa, qb)`, operations will be applied using `b`.
pub fn chain_tuple<B: UnitaryBuilder>(b: &mut B, qa: Qubit, qb: Qubit) -> DoubleQubitChain<B> {
    DoubleQubitChain::new(b, qa, qb)
}

/// Produce a chaining struct for a qubit vec `qs`, operations will be applied using `b`.
pub fn chain_vec<B: UnitaryBuilder>(b: &mut B, qs: Vec<Qubit>) -> VecQubitChain<B> {
    VecQubitChain::new(b, qs)
}

/// Chaining struct for a single qubit (which may have multiple indices)
#[derive(Debug)]
pub struct SingleQubitChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    q: Qubit,
}

/// Chaining struct for a pair of qubits (each may have multiple indices)
#[derive(Debug)]
pub struct DoubleQubitChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    qa: Qubit,
    qb: Qubit,
}

/// Chaining struct for a vector of qubits (each may have multiple indices)
#[derive(Debug)]
pub struct VecQubitChain<'a, B: UnitaryBuilder> {
    builder: &'a mut B,
    qs: Vec<Qubit>,
}

impl<'a, B: UnitaryBuilder> SingleQubitChain<'a, B> {
    /// Make a new SingleQubitChain. Prefer to use `chain`.
    pub fn new(builder: &'a mut B, q: Qubit) -> Self {
        SingleQubitChain::<'a, B> { builder, q }
    }
    /// Release the contained qubit.
    pub fn release(self) -> Qubit {
        self.q()
    }
    /// Release the contained qubit.
    pub fn q(self) -> Qubit {
        self.q
    }

    /// Split the qubit, select the given indices and transfer them to a new qubit, leave the
    /// remaining indices in another qubit. This uses the relative indices (0 refers to whatever the
    /// first index of the contained qubit is).
    pub fn split(self, indices: Vec<u64>) -> Result<DoubleQubitChain<'a, B>, &'static str> {
        let (qa, qb) = self.builder.split(self.q, indices)?;
        Ok(DoubleQubitChain::new(self.builder, qa, qb))
    }
    /// Split the qubit, select the given indices and transfer them to a new qubit, leave the
    /// remaining indices in another qubit. This uses the absolute indices (0 refers to the 0th
    /// absolute index, even if it isn't in the contained qubit, throwing an error).
    pub fn split_absolute(
        self,
        selected_indices: Vec<u64>,
    ) -> Result<DoubleQubitChain<'a, B>, &'static str> {
        let (qa, qb) = self.builder.split_absolute(self.q, selected_indices)?;
        Ok(DoubleQubitChain::new(self.builder, qa, qb))
    }
    /// Split each contained index into its own qubit object, returns a chaining struct for the vec
    /// of resulting qubits.
    pub fn split_all(self) -> Result<VecQubitChain<'a, B>, &'static str> {
        let qs = self.builder.split_all(self.q);
        Ok(VecQubitChain::new(self.builder, qs))
    }
    /// Apply a matrix operation to the contained qubit.
    pub fn apply_mat(self, name: &str, mat: Vec<Complex<f64>>) -> Result<Self, &'static str> {
        let q = self.builder.mat(name, self.q, mat)?;
        Ok(Self::new(self.builder, q))
    }
    /// Apply a sparse matrix operation to the contained qubit.
    pub fn apply_sparse_mat(
        self,
        name: &str,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Self, &'static str> {
        let q = self.builder.sparse_mat(name, self.q, mat, natural_order)?;
        Ok(Self::new(self.builder, q))
    }
    /// Apply the NOT operation to the contained qubit.
    pub fn not(self) -> Self {
        let q = self.builder.not(self.q);
        Self::new(self.builder, q)
    }
    /// Apply the X operation to the contained qubit.
    pub fn x(self) -> Self {
        let q = self.builder.x(self.q);
        Self::new(self.builder, q)
    }
    /// Apply the Y operation to the contained qubit.
    pub fn y(self) -> Self {
        let q = self.builder.y(self.q);
        Self::new(self.builder, q)
    }
    /// Apply the Z operation to the contained qubit.
    pub fn z(self) -> Self {
        let q = self.builder.z(self.q);
        Self::new(self.builder, q)
    }
    /// Apply a H to the contained qubit.
    pub fn hadamard(self) -> Self {
        let q = self.builder.hadamard(self.q);
        Self::new(self.builder, q)
    }
    /// Map the qubit by the given function.
    pub fn apply(self, f: impl FnOnce(&mut B, Qubit) -> Qubit) -> Self {
        let q = f(self.builder, self.q);
        Self::new(self.builder, q)
    }
    /// Map the qubit by the given function, resulting in two qubits.
    pub fn apply_cut(
        self,
        f: impl FnOnce(&mut B, Qubit) -> (Qubit, Qubit),
    ) -> DoubleQubitChain<'a, B> {
        let (qa, qb) = f(self.builder, self.q);
        DoubleQubitChain::new(self.builder, qa, qb)
    }
    /// Map the qubit by the given function, resulting in a vec of qubits.
    pub fn apply_split(self, f: impl FnOnce(&mut B, Qubit) -> Vec<Qubit>) -> VecQubitChain<'a, B> {
        let qs = f(self.builder, self.q);
        VecQubitChain::new(self.builder, qs)
    }
}

impl<'a, B: UnitaryBuilder> DoubleQubitChain<'a, B> {
    /// Make a new `DoubleQubitChain`, prefer to use `chain_tuple`.
    pub fn new(builder: &'a mut B, qa: Qubit, qb: Qubit) -> Self {
        DoubleQubitChain::<'a, B> { builder, qa, qb }
    }
    /// Release the contained qubit tuple
    pub fn release(self) -> (Qubit, Qubit) {
        self.qab()
    }
    /// Release the contained qubit tuple
    pub fn qab(self) -> (Qubit, Qubit) {
        (self.qa, self.qb)
    }
    /// Merge the contained qubit tuple into a single qubit, wrap in a chaining struct.
    pub fn merge(self) -> SingleQubitChain<'a, B> {
        let q = self.builder.merge(vec![self.qa, self.qb]);
        SingleQubitChain::new(self.builder, q)
    }
    /// Split all the indices for each qubit into their own qubits, returned the chained struct for
    /// the vec of qubits.
    pub fn split_all(self) -> VecQubitChain<'a, B> {
        let q = self.builder.merge(vec![self.qa, self.qb]);
        let qs = self.builder.split_all(q);
        VecQubitChain::new(self.builder, qs)
    }
    /// Apply a swap op to the contained qubits, will only succeed of the qubits are of equal size.
    pub fn swap(self) -> Result<Self, &'static str> {
        let (qa, qb) = self.builder.swap(self.qa, self.qb)?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Swap the positions of the contained qubits. This is not a quantum operation, rather a
    /// a bookkeeping one.
    pub fn physical_swap(self) -> Self {
        Self::new(self.builder, self.qb, self.qa)
    }
    /// Apply a function operation to the contained qubits, the first will act as the readin
    /// register and the second as the output.
    pub fn apply_function_op(
        self,
        f: impl Fn(u64) -> (u64, f64) + Send + Sync + 'static,
    ) -> Result<Self, &'static str> {
        let (qa, qb) = self.builder.apply_function(self.qa, self.qb, Box::new(f))?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Apply a function op which has been already boxed.
    pub fn apply_boxed_function_op(
        self,
        f: Box<Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<Self, &'static str> {
        let (qa, qb) = self.builder.apply_function(self.qa, self.qb, f)?;
        Ok(Self::new(self.builder, qa, qb))
    }
    /// Apply a function which outputs a single qubit.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Qubit, Qubit) -> Qubit,
    ) -> SingleQubitChain<'a, B> {
        let q = f(self.builder, self.qa, self.qb);
        SingleQubitChain::new(self.builder, q)
    }
    /// Apply a function which outputs a tuple of qubits.
    pub fn apply(self, f: impl FnOnce(&mut B, Qubit, Qubit) -> (Qubit, Qubit)) -> Self {
        let (qa, qb) = f(self.builder, self.qa, self.qb);
        Self::new(self.builder, qa, qb)
    }
    /// Apply a function which outputs a vector of qubits.
    pub fn apply_split(
        self,
        f: impl FnOnce(&mut B, Qubit, Qubit) -> Vec<Qubit>,
    ) -> VecQubitChain<'a, B> {
        let qs = f(self.builder, self.qa, self.qb);
        VecQubitChain::new(self.builder, qs)
    }
}

impl<'a, B: UnitaryBuilder> VecQubitChain<'a, B> {
    /// Make a new `VecQubitChain`, prefer to use `chain_vec`.
    pub fn new(builder: &'a mut B, qs: Vec<Qubit>) -> Self {
        VecQubitChain::<'a, B> { builder, qs }
    }
    /// Release the contained vec of qubits.
    pub fn release(self) -> Vec<Qubit> {
        self.qs()
    }
    /// Release the contained vec of qubits.
    pub fn qs(self) -> Vec<Qubit> {
        self.qs
    }
    /// Merge the contained vec of qubits into a single qubit.
    pub fn merge(self) -> SingleQubitChain<'a, B> {
        let q = self.builder.merge(self.qs);
        SingleQubitChain::new(self.builder, q)
    }
    /// Partition the contained qubits into two groups by their index in the underlying vector.
    /// Merge each group into a qubit and produce a chained struct for the tuple.
    pub fn partition_by_relative(
        self,
        f: impl Fn(u64) -> bool,
    ) -> Result<DoubleQubitChain<'a, B>, &'static str> {
        let (a, b): (Vec<_>, Vec<_>) = self
            .qs
            .into_iter()
            .enumerate()
            .partition(|(i, _)| f(*i as u64));

        if a.is_empty() {
            Err("Partition must provide at least one qubit to first entry.")
        } else if b.is_empty() {
            Err("Partition must provide at least one qubit to second entry.")
        } else {
            let f = |vs: Vec<(usize, Qubit)>| -> Vec<Qubit> {
                vs.into_iter().map(|(_, q)| q).collect()
            };
            let qa = self.builder.merge(f(a));
            let qb = self.builder.merge(f(b));
            Ok(DoubleQubitChain::new(self.builder, qa, qb))
        }
    }
    /// Flatten the qubits: make a chain struct representing the vec of all single-index
    /// qubits which can be made from the current set of owned indices.
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
    /// Apply a function which outputs a single qubit.
    pub fn apply_merge(
        self,
        f: impl FnOnce(&mut B, Vec<Qubit>) -> Qubit,
    ) -> SingleQubitChain<'a, B> {
        let q = f(self.builder, self.qs);
        SingleQubitChain::new(self.builder, q)
    }
    /// Apply a function which outputs a tuple of qubits.
    pub fn apply_partition(
        self,
        f: impl FnOnce(&mut B, Vec<Qubit>) -> (Qubit, Qubit),
    ) -> DoubleQubitChain<'a, B> {
        let (qa, qb) = f(self.builder, self.qs);
        DoubleQubitChain::new(self.builder, qa, qb)
    }
    /// Apply a function which outputs a vector of qubits.
    pub fn apply(self, f: impl FnOnce(&mut B, Vec<Qubit>) -> Vec<Qubit>) -> Self {
        let qs = f(self.builder, self.qs);
        Self::new(self.builder, qs)
    }
}
