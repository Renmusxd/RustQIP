use crate::errors::CircuitError;
use crate::pipeline::*;
use crate::qubits::*;
use crate::state_ops::*;
use crate::Complex;
use num::Zero;
use std::fmt;

/// A function which takes a builder, a qubit, and a set of measured values, and constructs a
/// circuit, outputting the resulting qubit.
type SingleQubitSideChannelFn =
    dyn Fn(&mut dyn UnitaryBuilder, Qubit, &[u64]) -> Result<Qubit, CircuitError>;

/// A function which takes a builder, a vec of qubits, and a set of measured values, and constructs a
/// circuit, outputting the resulting qubits.
type SideChannelFn =
    dyn Fn(&mut dyn UnitaryBuilder, Vec<Qubit>, &[u64]) -> Result<Vec<Qubit>, CircuitError>;

/// A function which takes a builder, a qubit with possibly extra indices, and a set of measured
/// values, and constructs a circuit, outputting the resulting qubit.
type SideChannelHelperFn =
    dyn Fn(&mut dyn UnitaryBuilder, Qubit, &[u64]) -> Result<Vec<Qubit>, CircuitError>;

/// A function to build rows of a sparse matrix. Takes a row and outputs columns and entries.
type SparseBuilderFn = dyn Fn(u64) -> Vec<(u64, Complex<f64>)>;

/// A builder which support unitary operations
pub trait UnitaryBuilder {
    // Things like X, Y, Z, NOT, H, SWAP, ... go here

    /// Build a builder which uses `q` as a condition.
    fn with_condition(&mut self, q: Qubit) -> ConditionalContextBuilder;

    /// Build a generic matrix op, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^(2n)).
    fn mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Complex<f64>>,
    ) -> Result<Qubit, CircuitError>;

    /// Build a matrix op from real numbers, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^(2n)).
    fn real_mat(&mut self, name: &str, q: Qubit, mat: &[f64]) -> Result<Qubit, CircuitError> {
        self.mat(name, q, from_reals(mat))
    }

    /// Build a sparse matrix op, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^n).
    fn sparse_mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Qubit, CircuitError>;

    /// Build a sparse matrix op from `f`, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^n).
    fn sparse_mat_from_fn(
        &mut self,
        name: &str,
        q: Qubit,
        f: Box<SparseBuilderFn>,
        natural_order: bool,
    ) -> Result<Qubit, CircuitError> {
        let n = q.indices.len();
        let mat = make_sparse_matrix_from_function(n, f, natural_order);
        self.sparse_mat(name, q, mat, false)
    }

    /// Build a sparse matrix op from real numbers, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^n).
    fn real_sparse_mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: &[Vec<(u64, f64)>],
        natural_order: bool,
    ) -> Result<Qubit, CircuitError> {
        let mat = mat
            .iter()
            .map(|v| {
                v.iter()
                    .cloned()
                    .map(|(c, r)| (c, Complex { re: r, im: 0.0 }))
                    .collect()
            })
            .collect();
        self.sparse_mat(name, q, mat, natural_order)
    }

    /// Apply NOT to `q`, if `q` is multiple indices, apply to each
    fn not(&mut self, q: Qubit) -> Qubit {
        self.real_mat("not", q, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply X to `q`, if `q` is multiple indices, apply to each
    fn x(&mut self, q: Qubit) -> Qubit {
        self.real_mat("X", q, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply Y to `q`, if `q` is multiple indices, apply to each
    fn y(&mut self, q: Qubit) -> Qubit {
        self.mat(
            "Y",
            q,
            from_tuples(&[(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]),
        )
        .unwrap()
    }

    /// Apply Z to `q`, if `q` is multiple indices, apply to each
    fn z(&mut self, q: Qubit) -> Qubit {
        self.real_mat("Z", q, &[1.0, 0.0, 0.0, -1.0]).unwrap()
    }

    /// Apply H to `q`, if `q` is multiple indices, apply to each
    fn hadamard(&mut self, q: Qubit) -> Qubit {
        let inv_sqrt = 1.0f64 / 2.0f64.sqrt();
        self.real_mat("H", q, &[inv_sqrt, inv_sqrt, inv_sqrt, -inv_sqrt])
            .unwrap()
    }

    /// Transforms `|psi>` to `e^{i*theta}|psi>`
    fn phase(&mut self, q: Qubit, theta: f64) -> Qubit {
        let phase = Complex { re: 0.0, im: theta }.exp();
        self.mat(
            "Phase",
            q,
            vec![phase, Complex::zero(), Complex::zero(), phase],
        )
        .unwrap()
    }

    /// Apply SWAP to `qa` and `qb`
    fn swap(&mut self, qa: Qubit, qb: Qubit) -> Result<(Qubit, Qubit), CircuitError> {
        let op = self.make_swap_op(&qa, &qb)?;
        let qa_indices = qa.indices.clone();

        let name = String::from("swap");
        let q = self.merge_with_op(vec![qa, qb], Some((name, op)));
        self.split_absolute(q, qa_indices)
    }

    /// Make an operation from the boxed function `f`. This maps c|`q_in`>|`q_out`> to
    /// c*e^i`theta`|`q_in`>|`q_out` ^ `indx`> where `indx` and `theta` are the outputs from the
    /// function `f(x) = (indx, theta)`
    fn apply_function(
        &mut self,
        q_in: Qubit,
        q_out: Qubit,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Qubit, Qubit), CircuitError>;

    /// Merge the qubits in `qs` into a single qubit.
    fn merge(&mut self, qs: Vec<Qubit>) -> Qubit {
        self.merge_with_op(qs, None)
    }

    /// Split the qubit `q` into two qubits, one with relative `indices` and one with the remaining.
    fn split(&mut self, q: Qubit, indices: Vec<u64>) -> Result<(Qubit, Qubit), CircuitError> {
        for indx in &indices {
            if *indx > q.n() {
                let message = format!(
                    "All indices for splitting must be below q.n={:?}, found indx={:?}",
                    q.n(),
                    *indx
                );
                return CircuitError::make_err(message);
            }
        }
        if indices.is_empty() {
            CircuitError::make_str_err("Indices must contain at least one index.")
        } else if indices.len() == q.indices.len() {
            CircuitError::make_str_err("Indices must leave at least one index.")
        } else {
            let selected_indices: Vec<u64> =
                indices.into_iter().map(|i| q.indices[i as usize]).collect();
            self.split_absolute(q, selected_indices)
        }
    }

    /// Split the qubit `q` into two qubits, one with `selected_indices` and one with the remaining.
    fn split_absolute(
        &mut self,
        q: Qubit,
        selected_indices: Vec<u64>,
    ) -> Result<(Qubit, Qubit), CircuitError>;

    /// Split the qubit into many qubits, each with the given set of indices.
    fn split_absolute_many(
        &mut self,
        q: Qubit,
        index_groups: Vec<Vec<u64>>,
    ) -> Result<(Vec<Qubit>, Option<Qubit>), CircuitError> {
        index_groups
            .into_iter()
            .try_fold((vec![], Some(q)), |(mut qs, q), indices| {
                if let Some(q) = q {
                    if q.indices == indices {
                        qs.push(q);
                        Ok((qs, None))
                    } else {
                        let (hq, tq) = self.split_absolute(q, indices)?;
                        qs.push(hq);
                        Ok((qs, Some(tq)))
                    }
                } else {
                    Ok((qs, None))
                }
            })
    }

    /// Split `q` into a single qubit for each index.
    fn split_all(&mut self, q: Qubit) -> Vec<Qubit> {
        if q.indices.len() == 1 {
            vec![q]
        } else {
            let mut indices: Vec<Vec<u64>> = q.indices.iter().cloned().map(|i| vec![i]).collect();
            indices.pop();
            // Cannot fail since all indices are from q.
            let (mut qs, q) = self.split_absolute_many(q, indices).unwrap();
            if let Some(q) = q {
                qs.push(q);
            };
            qs
        }
    }

    /// Build a generic matrix op.
    fn make_mat_op(
        &self,
        q: &Qubit,
        data: Vec<Complex<f64>>,
    ) -> Result<QubitOp, CircuitError> {
        make_matrix_op(q.indices.clone(), data)
    }

    /// Build a sparse matrix op
    fn make_sparse_mat_op(
        &self,
        q: &Qubit,
        data: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<QubitOp, CircuitError> {
        make_sparse_matrix_op(q.indices.clone(), data, natural_order)
    }

    /// Build a swap op. qa and qb must have the same number of indices.
    fn make_swap_op(&self, qa: &Qubit, qb: &Qubit) -> Result<QubitOp, CircuitError> {
        make_swap_op(qa.indices.clone(), qb.indices.clone())
    }

    /// Make a function op. f must be boxed so that this function doesn't need to be parameterized.
    fn make_function_op(
        &self,
        q_in: &Qubit,
        q_out: &Qubit,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<QubitOp, CircuitError> {
        make_function_op(q_in.indices.clone(), q_out.indices.clone(), f)
    }

    /// Merge qubits using a generic state processing function.
    fn merge_with_op(&mut self, qs: Vec<Qubit>, named_operator: Option<(String, QubitOp)>)
        -> Qubit;

    /// Measure all qubit states and probabilities, does not edit state (thus Unitary). Returns
    /// qubit and handle.
    fn stochastic_measure(&mut self, q: Qubit) -> (Qubit, u64);

    /// Create a circuit portion which depends on the classical results of measuring some qubits.
    fn single_qubit_classical_sidechannel(
        &mut self,
        q: Qubit,
        handles: &[MeasurementHandle],
        f: Box<SingleQubitSideChannelFn>,
    ) -> Qubit {
        self.classical_sidechannel(
            vec![q],
            handles,
            Box::new(
                move |b: &mut dyn UnitaryBuilder,
                      mut qs: Vec<Qubit>,
                      measurements: &[u64]|
                      -> Result<Vec<Qubit>, CircuitError> {
                    Ok(vec![f(b, qs.pop().unwrap(), measurements)?])
                },
            ),
        )
        .pop()
        .unwrap()
    }

    /// Create a circuit portion which depends on the classical results of measuring some qubits.
    fn classical_sidechannel(
        &mut self,
        qs: Vec<Qubit>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelFn>,
    ) -> Vec<Qubit> {
        let index_groups: Vec<_> = qs.iter().map(|q| &q.indices).cloned().collect();
        let f = Box::new(
            move |b: &mut dyn UnitaryBuilder,
                  q: Qubit,
                  ms: &[u64]|
                  -> Result<Vec<Qubit>, CircuitError> {
                let (qs, _) = b.split_absolute_many(q, index_groups.clone())?;
                f(b, qs, ms)
            },
        );
        self.sidechannel_helper(qs, handles, f)
    }

    /// A helper function for the classical_sidechannel. Takes a set of qubits to pass to the
    /// subcircuit, a set of handles whose measured values will also be passed, and a function
    /// which matches to description of `SideChannelHelperFn`. Returns a set of qubits whose indices
    /// match those of the input qubits.
    /// This shouldn't be called in circuits, and is a helper to `classical_sidechannel` and
    /// `single_qubit_classical_sidechannel`.
    fn sidechannel_helper(
        &mut self,
        qs: Vec<Qubit>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Qubit>;
}

/// Helper function for Boxing static functions and applying using the given UnitaryBuilder.
pub fn apply_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    q_in: Qubit,
    q_out: Qubit,
    f: F,
) -> Result<(Qubit, Qubit), CircuitError> {
    b.apply_function(q_in, q_out, Box::new(f))
}

/// Helper function for Boxing static functions and building sparse mats using the given
/// UnitaryBuilder.
pub fn apply_sparse_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    q_in: Qubit,
    q_out: Qubit,
    f: F,
) -> Result<(Qubit, Qubit), CircuitError> {
    b.apply_function(q_in, q_out, Box::new(f))
}

/// A basic builder for unitary and non-unitary ops.
#[derive(Default, Debug)]
pub struct OpBuilder {
    qubit_index: u64,
    op_id: u64,
    temp_zero_qubits: Vec<Qubit>,
    temp_one_qubits: Vec<Qubit>,
}

impl OpBuilder {
    /// Build a new OpBuilder
    pub fn new() -> OpBuilder {
        OpBuilder::default()
    }

    /// Build a new qubit with `n` indices
    pub fn qubit(&mut self, n: u64) -> Result<Qubit, CircuitError> {
        if n == 0 {
            CircuitError::make_str_err("Qubit n must be greater than 0.")
        } else {
            let base_index = self.qubit_index;
            self.qubit_index += n;
            Qubit::new(self.get_op_id(), (base_index..self.qubit_index).collect())
        }
    }

    /// Builds a vector of new qubits
    pub fn qubits(&mut self, ns: &[u64]) -> Result<Vec<Qubit>, CircuitError> {
        ns.iter()
            .try_for_each(|n| {
                if *n == 0 {
                    CircuitError::make_str_err("Qubit n must be greater than 0.")
                } else {
                    Ok(())
                }
            })
            .map(|_| ns.iter().map(|n| self.qubit(*n).unwrap()).collect())
    }

    /// If you just plan to call unwrap this is cleaner.
    pub fn q(&mut self, n: u64) -> Qubit {
        self.qubit(n).unwrap()
    }

    /// Build a new qubit with `n` indices, return it plus a handle which can be
    /// used for feeding in an initial state.
    pub fn qubit_and_handle(&mut self, n: u64) -> Result<(Qubit, QubitHandle), CircuitError> {
        let q = self.qubit(n)?;
        let h = q.handle();
        Ok((q, h))
    }

    /// Get a temporary qubit with value `|0n>` or `|1n>`.
    /// This value is not checked and may be subject to the noise of your circuit, since it can
    /// recycle qubits which were returned with `return_temp_qubit`. If not enough qubits have been
    /// returned, then new qubits may be allocated (and initialized with the correct value).
    pub fn get_temp_qubit(&mut self, n: u64, value: bool) -> Qubit {
        let (op_vec, other_vec) = if value {
            (&mut self.temp_one_qubits, &mut self.temp_zero_qubits)
        } else {
            (&mut self.temp_zero_qubits, &mut self.temp_one_qubits)
        };
        let mut acquired_qubits = op_vec.split_off(n as usize);

        // If we didn't get enough, take from temps with the wrong bit.
        if acquired_qubits.len() < n as usize {
            let remaining = n - acquired_qubits.len() as u64;
            let additional_qubits = other_vec.split_off(remaining as usize);
            let q = self.merge(additional_qubits);
            let q = self.not(q);
            acquired_qubits.extend(self.split_all(q).into_iter());
        }

        // If there still aren't enough, start allocating more qubits (and apply NOT if needed).
        if acquired_qubits.len() < n as usize {
            let remaining = n - acquired_qubits.len() as u64;
            let q = self.qubit(remaining).unwrap();
            let q = if value { self.not(q) } else { q };
            acquired_qubits.push(q);
        };

        // Make the temp qubit.
        self.merge(acquired_qubits)
    }

    /// Return a temporary qubit which is supposed to have a given value `|0n>` or `|1n>`
    /// This value is not checked and may be subject to the noise of your circuit, in turn causing
    /// noise to future calls to `get_temp_qubit`.
    pub fn return_temp_qubit(&mut self, q: Qubit, value: bool) {
        let qs = self.split_all(q);
        let op_vec = if value {
            &mut self.temp_one_qubits
        } else {
            &mut self.temp_zero_qubits
        };
        op_vec.extend(qs.into_iter());
    }

    /// Add a measure op to the pipeline for `q` and return a reference which can
    /// later be used to access the measured value from the results of `pipeline::run`.
    pub fn measure(&mut self, q: Qubit) -> (Qubit, MeasurementHandle) {
        self.measure_basis(q, 0.0)
    }

    /// Measure in the basis of `cos(phase)|0> + sin(phase)|1>`
    pub fn measure_basis(&mut self, q: Qubit, angle: f64) -> (Qubit, MeasurementHandle) {
        let id = self.get_op_id();
        let modifier = StateModifier::new_measurement_basis(
            String::from("measure"),
            id,
            q.indices.clone(),
            angle,
        );
        let modifier = Some(modifier);
        let q = Qubit::merge_with_modifier(id, vec![q], modifier);
        Qubit::make_measurement_handle(self.get_op_id(), q)
    }

    fn get_op_id(&mut self) -> u64 {
        let tmp = self.op_id;
        self.op_id += 1;
        tmp
    }
}

impl UnitaryBuilder for OpBuilder {
    fn with_condition(&mut self, q: Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: Some(q),
        }
    }

    fn mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Complex<f64>>,
    ) -> Result<Qubit, CircuitError> {
        // Special case for broadcasting ops
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs
                .into_iter()
                .map(|q| self.mat(name, q, mat.clone()).unwrap())
                .collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let op = self.make_mat_op(&q, mat)?;
            let name = String::from(name);
            Ok(self.merge_with_op(vec![q], Some((name, op))))
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Qubit, CircuitError> {
        // Special case for broadcasting ops
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs
                .into_iter()
                .map(|q| {
                    self.sparse_mat(name, q, mat.clone(), natural_order)
                        .unwrap()
                })
                .collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let op = self.make_sparse_mat_op(&q, mat, natural_order)?;
            let name = String::from(name);
            Ok(self.merge_with_op(vec![q], Some((name, op))))
        }
    }

    fn apply_function(
        &mut self,
        q_in: Qubit,
        q_out: Qubit,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Qubit, Qubit), CircuitError> {
        let op = self.make_function_op(&q_in, &q_out, f)?;
        let in_indices = q_in.indices.clone();
        let name = String::from("f");
        let q = self.merge_with_op(vec![q_in, q_out], Some((name, op)));
        self.split_absolute(q, in_indices)
    }

    fn split_absolute(
        &mut self,
        q: Qubit,
        selected_indices: Vec<u64>,
    ) -> Result<(Qubit, Qubit), CircuitError> {
        Qubit::split_absolute(self.get_op_id(), self.get_op_id(), q, selected_indices)
    }

    fn merge_with_op(
        &mut self,
        qs: Vec<Qubit>,
        named_operator: Option<(String, QubitOp)>,
    ) -> Qubit {
        let modifier = named_operator.map(|(name, op)| StateModifier::new_unitary(name, op));
        Qubit::merge_with_modifier(self.get_op_id(), qs, modifier)
    }

    fn stochastic_measure(&mut self, q: Qubit) -> (Qubit, u64) {
        let id = self.get_op_id();
        let modifier = StateModifier::new_stochastic_measurement(
            String::from("stochastic"),
            id,
            q.indices.clone(),
        );
        let modifier = Some(modifier);
        let q = Qubit::merge_with_modifier(id, vec![q], modifier);
        (q, id)
    }

    fn sidechannel_helper(
        &mut self,
        qs: Vec<Qubit>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Qubit> {
        let index_groups: Vec<_> = qs.iter().map(|q| &q.indices).cloned().collect();
        let req_qubits = self.qubit_index + 1;

        let f = Box::new(
            move |measurements: &[u64]| -> Result<Vec<StateModifier>, CircuitError> {
                let mut b = Self::new();
                let q = b.qubit(req_qubits)?;
                let qs = f(&mut b, q, measurements)?;
                let q = b.merge(qs);
                Ok(get_owned_opfns(q))
            },
        );

        let modifier = Some(StateModifier::new_side_channel(
            String::from("SideInputCircuit"),
            handles,
            f,
        ));
        let q = Qubit::merge_with_modifier(self.get_op_id(), qs, modifier);

        let deps = handles.iter().map(|m| m.clone_qubit()).collect();
        let q = Qubit::add_deps(q, deps);
        let (qs, _) = self.split_absolute_many(q, index_groups).unwrap();
        qs
    }
}

/// An op builder which depends on the value of a given qubit (COPs)
pub struct ConditionalContextBuilder<'a> {
    parent_builder: &'a mut dyn UnitaryBuilder,
    conditioned_qubit: Option<Qubit>,
}

impl<'a> fmt::Debug for ConditionalContextBuilder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ConditionalContextBuilder({:?})", self.conditioned_qubit)
    }
}

impl<'a> ConditionalContextBuilder<'a> {
    /// Release the qubit used to build this builder
    pub fn release_qubit(self: Self) -> Qubit {
        match self.conditioned_qubit {
            Some(q) => q,
            None => panic!("Conditional context builder failed to populate qubit."),
        }
    }

    fn get_conditional_qubit(&mut self) -> Qubit {
        self.conditioned_qubit.take().unwrap()
    }

    fn set_conditional_qubit(&mut self, cq: Qubit) {
        self.conditioned_qubit = Some(cq);
    }
}

impl<'a> UnitaryBuilder for ConditionalContextBuilder<'a> {
    fn with_condition(&mut self, q: Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: Some(q),
        }
    }

    fn mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Complex<f64>>,
    ) -> Result<Qubit, CircuitError> {
        // Special case for applying mat to each qubit in collection.
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs
                .into_iter()
                .map(|q| self.mat(name, q, mat.clone()).unwrap())
                .collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let op = self.make_mat_op(&q, mat)?;
            let cq = self.get_conditional_qubit();
            let cq_indices = cq.indices.clone();
            let name = format!("C({})", name);
            let q = self.merge_with_op(vec![cq, q], Some((name, op)));
            let (cq, q) = self.split_absolute(q, cq_indices).unwrap();

            self.set_conditional_qubit(cq);
            Ok(q)
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        q: Qubit,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Qubit, CircuitError> {
        // Special case for applying mat to each qubit in collection.
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs
                .into_iter()
                .map(|q| {
                    self.sparse_mat(name, q, mat.clone(), natural_order)
                        .unwrap()
                })
                .collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let op = self.make_sparse_mat_op(&q, mat, natural_order)?;
            let cq = self.get_conditional_qubit();
            let cq_indices = cq.indices.clone();
            let name = format!("C({})", name);
            let q = self.merge_with_op(vec![cq, q], Some((name, op)));
            let (cq, q) = self.split_absolute(q, cq_indices).unwrap();

            self.set_conditional_qubit(cq);
            Ok(q)
        }
    }

    fn swap(&mut self, qa: Qubit, qb: Qubit) -> Result<(Qubit, Qubit), CircuitError> {
        let op = self.make_swap_op(&qa, &qb)?;
        let cq = self.get_conditional_qubit();
        let cq_indices = cq.indices.clone();
        let qa_indices = qa.indices.clone();
        let name = String::from("C(swap)");
        let q = self.merge_with_op(vec![cq, qa, qb], Some((name, op)));
        let (cq, q) = self.split_absolute(q, cq_indices).unwrap();
        let (qa, qb) = self.split_absolute(q, qa_indices).unwrap();

        self.set_conditional_qubit(cq);
        Ok((qa, qb))
    }

    fn apply_function(
        &mut self,
        q_in: Qubit,
        q_out: Qubit,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Qubit, Qubit), CircuitError> {
        let op = self.make_function_op(&q_in, &q_out, f)?;
        let cq = self.get_conditional_qubit();

        let cq_indices = cq.indices.clone();
        let in_indices = q_in.indices.clone();
        let name = String::from("C(f)");
        let q = self.merge_with_op(vec![cq, q_in, q_out], Some((name, op)));
        let (cq, q) = self.split_absolute(q, cq_indices).unwrap();
        let (q_in, q_out) = self.split_absolute(q, in_indices).unwrap();

        self.set_conditional_qubit(cq);
        Ok((q_in, q_out))
    }

    fn split_absolute(
        &mut self,
        q: Qubit,
        selected_indices: Vec<u64>,
    ) -> Result<(Qubit, Qubit), CircuitError> {
        self.parent_builder.split_absolute(q, selected_indices)
    }

    fn make_mat_op(
        &self,
        q: &Qubit,
        data: Vec<Complex<f64>>,
    ) -> Result<QubitOp, CircuitError> {
        match &self.conditioned_qubit {
            Some(cq) => make_control_op(
                cq.indices.clone(),
                self.parent_builder.make_mat_op(q, data)?,
            ),
            None => panic!("Conditional context builder failed to populate qubit."),
        }
    }

    fn make_sparse_mat_op(
        &self,
        q: &Qubit,
        data: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<QubitOp, CircuitError> {
        match &self.conditioned_qubit {
            Some(cq) => make_control_op(
                cq.indices.clone(),
                self.parent_builder
                    .make_sparse_mat_op(q, data, natural_order)?,
            ),
            None => panic!("Conditional context builder failed to populate qubit."),
        }
    }

    fn make_swap_op(&self, qa: &Qubit, qb: &Qubit) -> Result<QubitOp, CircuitError> {
        match &self.conditioned_qubit {
            Some(cq) => {
                let op = self.parent_builder.make_swap_op(qa, qb)?;
                make_control_op(cq.indices.clone(), op)
            }
            None => panic!("Conditional context builder failed to populate qubit."),
        }
    }

    fn make_function_op(
        &self,
        q_in: &Qubit,
        q_out: &Qubit,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<QubitOp, CircuitError> {
        match &self.conditioned_qubit {
            Some(cq) => {
                let op = self.parent_builder.make_function_op(q_in, q_out, f)?;
                make_control_op(cq.indices.clone(), op)
            }
            None => panic!("Conditional context builder failed to populate qubit."),
        }
    }

    fn merge_with_op(
        &mut self,
        qs: Vec<Qubit>,
        named_operator: Option<(String, QubitOp)>,
    ) -> Qubit {
        self.parent_builder.merge_with_op(qs, named_operator)
    }

    fn stochastic_measure(&mut self, q: Qubit) -> (Qubit, u64) {
        self.parent_builder.stochastic_measure(q)
    }

    fn sidechannel_helper(
        &mut self,
        qs: Vec<Qubit>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Qubit> {
        let cq = self.get_conditional_qubit();
        let conditioned_indices = cq.indices.clone();
        let cindices_clone = conditioned_indices.clone();
        let f = Box::new(
            move |b: &mut dyn UnitaryBuilder,
                  q: Qubit,
                  ms: &[u64]|
                  -> Result<Vec<Qubit>, CircuitError> {
                let (cq, q) = b.split_absolute(q, conditioned_indices.clone())?;
                let mut b = b.with_condition(cq);
                f(&mut b, q, ms)
            },
        );
        let qs = self.parent_builder.sidechannel_helper(qs, handles, f);
        let index_groups: Vec<_> = qs.iter().map(|q| q.indices.clone()).collect();
        let q = self.merge(qs);
        let q = self.merge(vec![cq, q]);
        let (cq, q) = self.split_absolute(q, cindices_clone).unwrap();
        self.set_conditional_qubit(cq);
        let (qs, _) = self.split_absolute_many(q, index_groups).unwrap();
        qs
    }
}
