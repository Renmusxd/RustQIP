extern crate num;
extern crate rayon;

use std::cmp::max;
use std::collections::HashMap;
use std::collections::{BinaryHeap, VecDeque};

use num::complex::Complex;
use rayon::prelude::*;

use self::num::{One, Zero};
use crate::measurement_ops::{
    measure, measure_prob, measure_probs, prob_magnitude, soft_measure, MeasuredCondition,
};
use crate::qubits::*;
use crate::state_ops::*;
use crate::types::Precision;
use crate::utils;
use crate::utils::flip_bits;
use std::rc::Rc;

pub type SideChannelModifierFn = dyn Fn(&[u64]) -> Result<Vec<StateModifier>, &'static str>;

pub enum StateModifierType {
    UnitaryOp(QubitOp),
    MeasureState(u64, Vec<u64>, f64),
    StochasticMeasureState(u64, Vec<u64>, f64),
    SideChannelModifiers(Vec<MeasurementHandle>, Box<SideChannelModifierFn>),
}

pub struct StateModifier {
    name: String,
    modifier: StateModifierType,
}

impl StateModifier {
    pub fn new_unitary(name: String, op: QubitOp) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::UnitaryOp(op),
        }
    }

    pub fn new_measurement(name: String, id: u64, indices: Vec<u64>) -> StateModifier {
        StateModifier::new_measurement_basis(name, id, indices, 0.0)
    }

    pub fn new_measurement_basis(
        name: String,
        id: u64,
        indices: Vec<u64>,
        angle: f64,
    ) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::MeasureState(id, indices, angle),
        }
    }

    pub fn new_stochastic_measurement(name: String, id: u64, indices: Vec<u64>) -> StateModifier {
        StateModifier::new_stochastic_measurement_basis(name, id, indices, 0.0)
    }

    pub fn new_stochastic_measurement_basis(
        name: String,
        id: u64,
        indices: Vec<u64>,
        angle: f64,
    ) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::StochasticMeasureState(id, indices, angle),
        }
    }

    pub fn new_side_channel(
        name: String,
        handles: &[MeasurementHandle],
        f: Box<SideChannelModifierFn>,
    ) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::SideChannelModifiers(handles.to_vec(), f),
        }
    }
}

pub struct MeasurementHandle {
    pub qubit: Rc<Qubit>,
}

impl MeasurementHandle {
    pub fn get_id(&self) -> u64 {
        self.qubit.id
    }
}

impl Clone for MeasurementHandle {
    fn clone(&self) -> Self {
        MeasurementHandle {
            qubit: self.qubit.clone(),
        }
    }
}

impl std::cmp::Eq for MeasurementHandle {}

impl std::cmp::PartialEq for MeasurementHandle {
    fn eq(&self, other: &MeasurementHandle) -> bool {
        self.qubit == other.qubit
    }
}

impl std::cmp::Ord for MeasurementHandle {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.qubit.cmp(&other.qubit)
    }
}

impl std::cmp::PartialOrd for MeasurementHandle {
    fn partial_cmp(&self, other: &MeasurementHandle) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
pub struct MeasuredResults<P: Precision> {
    results: HashMap<u64, (u64, P)>,
    stochastic_results: HashMap<u64, Vec<P>>,
}

impl<P: Precision> MeasuredResults<P> {
    pub fn new() -> MeasuredResults<P> {
        MeasuredResults::default()
    }

    /// Retrieve the measurement and likelihood for a given handle.
    pub fn get_measurement(&self, handle: &MeasurementHandle) -> Option<(u64, P)> {
        self.results.get(&handle.get_id()).cloned()
    }

    /// Clone the stochastic set of measurements for a given handle.
    pub fn clone_stochastic_measurements(&self, handle: u64) -> Option<Vec<P>> {
        self.stochastic_results.get(&handle).cloned()
    }

    /// Remove the set of measurements from the MeasuredResults struct, and return it.
    pub fn pop_stochastic_measurements(&mut self, handle: u64) -> Option<Vec<P>> {
        self.stochastic_results.remove(&handle)
    }
}

/// A trait which represents the state of the qubits
pub trait QuantumState<P: Precision> {
    /// Make new state with n qubits
    fn new(n: u64) -> Self;

    /// Initialize new state with initial states.
    fn new_from_initial_states(n: u64, states: &[QubitInitialState<P>]) -> Self;

    /// Get number of qubits represented by this state.
    fn n(&self) -> u64;

    /// Function to mutate self into the state with op applied.
    fn apply_op(&mut self, op: &QubitOp) {
        self.apply_op_with_name(None, op)
    }

    /// Apply op with a given name. Mutate self using op.
    fn apply_op_with_name(&mut self, name: Option<&str>, op: &QubitOp);

    /// Mutate self with measurement, return result as index and probability
    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P);

    /// Perform calculations of `measure` without mutating result. Returns a possible measured value
    /// and associated probability.
    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, P);

    /// Give the total magnitude represented by this state. Most often 1.0
    fn state_magnitude(&self) -> P;

    /// Measure stochastically, do not alter internal state.
    /// Returns a vector of size 2^indices.len()
    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P>;

    /// Consume the QuantumState object and return the state as a vector of complex numbers.
    /// `natural_order` means that qubit with index 0 is the least significant index bit, otherwise
    /// it's the largest.
    fn get_state(self, natural_order: bool) -> Vec<Complex<P>>;
}

/// A basic representation of a quantum state, given by a vector of complex numbers stored
/// locally on the machine (plus an arena of equal size to work in).
pub struct LocalQuantumState<P: Precision> {
    // A bundle with the quantum state data.
    pub n: u64,
    pub state: Vec<Complex<P>>,
    arena: Vec<Complex<P>>,
    multithread: bool,
}

impl<P: Precision> LocalQuantumState<P> {
    /// Build a local state using a set of initial states for subsets of the qubits.
    /// These initial states are made from the qubit handles.
    fn new_from_initial_states_and_multithread(
        n: u64,
        states: &[QubitInitialState<P>],
        multithread: bool,
    ) -> LocalQuantumState<P> {
        let max_init_n = states
            .iter()
            .map(|(indices, _)| indices)
            .cloned()
            .flatten()
            .max()
            .map(|m| m + 1);

        let n = max_init_n.map(|m| max(n, m)).unwrap_or(n);

        let mut cvec: Vec<Complex<P>> = (0..1 << n).map(|_| Complex::default()).collect();

        // Assume that all unrepresented indices are in the |0> state.
        let n_fullindices: u64 = states
            .iter()
            .map(|(indices, state)| match state {
                InitialState::FullState(_) => indices.len() as u64,
                _ => 0,
            })
            .sum();

        // Make the index template/base
        let template: u64 = states.iter().fold(0, |acc, (indices, state)| -> u64 {
            match state {
                InitialState::Index(val_indx) => sub_to_full(n, indices, *val_indx, acc),
                _ => acc,
            }
        });

        // Go through each combination of full index locations
        (0..1 << n_fullindices).for_each(|i| {
            // Calculate the offset from template, and the product of fullstates.
            let (delta_index, _, val) =
                states
                    .iter()
                    .fold((0u64, 0u64, Complex::one()), |acc, (indices, state)| {
                        if let InitialState::FullState(vals) = state {
                            let (superindex_acc, sub_index_offset, val_acc) = acc;
                            // Now we need to make additions to the superindex by adding bits based on
                            // indices, as well as return the value given by the [sub .. sub + len] bits
                            // from i.
                            let index_mask = (1 << indices.len() as u64) - 1;
                            let val_index_bits = (i >> sub_index_offset) & index_mask;
                            let val_acc = val_acc * vals[val_index_bits as usize];

                            let superindex_delta: u64 = indices
                                .iter()
                                .enumerate()
                                .map(|(j, indx)| {
                                    let bit = (val_index_bits >> j as u64) & 1u64;
                                    bit << (n - 1 - indx)
                                })
                                .sum();
                            (
                                superindex_acc + superindex_delta,
                                sub_index_offset + indices.len() as u64,
                                val_acc,
                            )
                        } else {
                            acc
                        }
                    });
            cvec[(delta_index + template) as usize] = val;
        });

        let arena = vec![Complex::zero(); cvec.len()];
        LocalQuantumState {
            n,
            state: cvec.clone(),
            arena,
            multithread,
        }
    }

    pub fn new_from_full_state(
        n: u64,
        state: Vec<Complex<P>>,
        natural_order: bool,
        multithread: bool,
    ) -> Result<LocalQuantumState<P>, &'static str> {
        if state.len() != 1 << n as usize {
            return Err("State is not correct size");
        }

        let arena = vec![Complex::zero(); state.len()];

        let state = if natural_order {
            let mut state: Vec<_> = state.into_iter().enumerate().collect();
            state.sort_by_key(|(indx, _)| flip_bits(n as usize, *indx as u64));
            state.into_iter().map(|(_, c)| c).collect()
        } else {
            state
        };

        Ok(LocalQuantumState {
            n,
            state,
            arena,
            multithread,
        })
    }

    /// Clone the state in either the `natural_order` or the internal order.
    pub fn clone_state(&mut self, natural_order: bool) -> Vec<Complex<P>> {
        if natural_order {
            let n = self.n;
            let state = &self.state;
            let f = |(i, outputloc): (usize, &mut Complex<P>)| {
                *outputloc = state[utils::flip_bits(n as usize, i as u64) as usize];
            };

            if self.multithread {
                self.arena.par_iter_mut().enumerate().for_each(f);
            } else {
                self.arena.iter_mut().enumerate().for_each(f);
            }
            self.arena.clone()
        } else {
            self.state.clone()
        }
    }

    /// Rotate to a new computational basis:
    /// `|0'> =  cos(angle)|0> + sin(angle)|1>`
    /// `|1'> = -sin(angle)|0> + cos(angle)|1>`
    pub fn rotate_basis(&mut self, indices: &[u64], angle: f64) {
        if angle != 0.0 {
            let (sangle, cangle) = angle.sin_cos();
            let basis_mat = from_reals(&[cangle, -sangle, sangle, cangle]);
            indices.iter().for_each(|indx| {
                let op = make_matrix_op(vec![*indx], basis_mat.clone()).unwrap();
                self.apply_op(&op);
            });
        }
    }

    pub fn set_multithreading(&mut self, multithread: bool) {
        self.multithread = multithread;
    }
}

impl<P: Precision> Clone for LocalQuantumState<P> {
    fn clone(&self) -> Self {
        LocalQuantumState {
            n: self.n,
            state: self.state.clone(),
            arena: self.arena.clone(),
            multithread: self.multithread,
        }
    }
}

pub enum InitialState<P: Precision> {
    FullState(Vec<Complex<P>>),
    Index(u64),
}

pub type QubitInitialState<P> = (Vec<u64>, InitialState<P>);

impl<P: Precision> QuantumState<P> for LocalQuantumState<P> {
    /// Build a new LocalQuantumState
    fn new(n: u64) -> LocalQuantumState<P> {
        LocalQuantumState::new_from_initial_states(n, &[])
    }

    /// Build a local state using a set of initial states for subsets of the qubits.
    /// These initial states are made from the qubit handles.
    fn new_from_initial_states(n: u64, states: &[QubitInitialState<P>]) -> LocalQuantumState<P> {
        Self::new_from_initial_states_and_multithread(n, states, true)
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn apply_op_with_name(&mut self, _name: Option<&str>, op: &QubitOp) {
        apply_op(
            self.n,
            op,
            &self.state,
            &mut self.arena,
            0,
            0,
            self.multithread,
        );
        std::mem::swap(&mut self.state, &mut self.arena);
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P) {
        self.rotate_basis(indices, angle);
        let measured_result = measure(
            self.n,
            indices,
            &self.state,
            &mut self.arena,
            None,
            measured,
            self.multithread,
        );
        self.rotate_basis(indices, -angle);

        std::mem::swap(&mut self.state, &mut self.arena);
        measured_result
    }

    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, P) {
        self.rotate_basis(indices, angle);
        let m = if let Some(m) = measured {
            m
        } else {
            soft_measure(self.n, indices, &self.state, None, self.multithread)
        };
        let p = measure_prob(self.n, m, indices, &self.state, None, self.multithread);
        self.rotate_basis(indices, -angle);
        (m, p)
    }

    fn state_magnitude(&self) -> P {
        prob_magnitude(&self.state, self.multithread)
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        self.rotate_basis(indices, angle);
        let probs = measure_probs(self.n, indices, &self.state, None, self.multithread);
        self.rotate_basis(indices, -angle);
        probs
    }

    fn get_state(mut self, natural_order: bool) -> Vec<Complex<P>> {
        if natural_order {
            let n = self.n;
            let state = self.state;
            let f = |(i, outputloc): (usize, &mut Complex<P>)| {
                *outputloc = state[utils::flip_bits(n as usize, i as u64) as usize];
            };

            if self.multithread {
                self.arena.par_iter_mut().enumerate().for_each(f);
            } else {
                self.arena.iter_mut().enumerate().for_each(f);
            }
            self.arena
        } else {
            self.state
        }
    }
}

/// Apply an QubitOp to the state `s` and return the new state.
fn fold_modify_state<P: Precision, QS: QuantumState<P>>(
    acc: (QS, MeasuredResults<P>),
    modifier: &StateModifier,
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    let (mut s, mut mr) = acc;
    match &modifier.modifier {
        StateModifierType::UnitaryOp(op) => {
            s.apply_op_with_name(Some(&modifier.name), op);
            Ok((s, mr))
        }
        StateModifierType::MeasureState(id, indices, angle) => {
            let result = s.measure(indices, None, *angle);
            mr.results.insert(id.clone(), result);
            Ok((s, mr))
        }
        StateModifierType::StochasticMeasureState(id, indices, angle) => {
            let result = s.stochastic_measure(indices, *angle);
            mr.stochastic_results.insert(id.clone(), result);
            Ok((s, mr))
        }
        StateModifierType::SideChannelModifiers(handles, f) => {
            let measured_values: Vec<_> = handles
                .iter()
                .map(|handle| mr.get_measurement(handle))
                .collect();
            measured_values.iter().try_for_each(|x| match x {
                Some(_) => Ok(()),
                None => Err("Not all measurements found"),
            })?;
            let measured_values: Vec<_> = measured_values
                .into_iter()
                .map(|m| m.map(|(m, _)| m).unwrap())
                .collect();
            let modifiers = f(&measured_values)?;
            modifiers.iter().try_fold((s, mr), fold_modify_state)
        }
    }
}

pub fn get_required_state_size_from_frontier(frontier: &[&Qubit]) -> u64 {
    frontier
        .iter()
        .map(|q| &q.indices)
        .cloned()
        .flatten()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

pub fn get_required_state_size<P: Precision>(
    frontier: &[&Qubit],
    states: &[QubitInitialState<P>],
) -> u64 {
    let max_qubit_n = get_required_state_size_from_frontier(frontier);
    let max_init_n = states
        .iter()
        .map(|(indices, _)| indices)
        .cloned()
        .flatten()
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    max(max_init_n, max_qubit_n)
}

/// Builds a default state of size `n`
pub fn run<P: Precision, QS: QuantumState<P>>(
    q: &Qubit,
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    run_with_statebuilder(q, |qs| -> Result<QS, &'static str> {
        let n = get_required_state_size_from_frontier(&qs);
        Ok(QS::new(n))
    })
}

pub fn run_with_init<P: Precision, QS: QuantumState<P>>(
    q: &Qubit,
    states: &[QubitInitialState<P>],
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    run_with_statebuilder(q, |qs| -> Result<QS, &'static str> {
        let n = get_required_state_size(&qs, states);
        Ok(QS::new_from_initial_states(n, states))
    })
}

pub fn run_with_statebuilder<
    P: Precision,
    QS: QuantumState<P>,
    F: FnOnce(Vec<&Qubit>) -> Result<QS, &'static str>,
>(
    q: &Qubit,
    state_builder: F,
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    let (frontier, ops) = get_opfns_and_frontier(q);
    let state = state_builder(frontier)?;
    run_with_state_and_ops(&ops, state)
}

pub fn run_with_state<P: Precision, QS: QuantumState<P>>(
    q: &Qubit,
    state: QS,
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    let (frontier, ops) = get_opfns_and_frontier(q);

    let req_n = get_required_state_size::<P>(&frontier, &[]);

    if req_n != state.n() {
        Err("Provided state n is not the required value for this circuit.")
    } else {
        run_with_state_and_ops(&ops, state)
    }
}

/// `run` the pipeline using `LocalQuantumState`.
pub fn run_local<P: Precision>(
    q: &Qubit,
) -> Result<(LocalQuantumState<P>, MeasuredResults<P>), &'static str> {
    run(q)
}

/// `run_with_init` the pipeline using `LocalQuantumState`
pub fn run_local_with_init<P: Precision>(
    q: &Qubit,
    states: &[QubitInitialState<P>],
) -> Result<(LocalQuantumState<P>, MeasuredResults<P>), &'static str> {
    run_with_init(q, states)
}

fn run_with_state_and_ops<P: Precision, QS: QuantumState<P>>(
    ops: &[&StateModifier],
    state: QS,
) -> Result<(QS, MeasuredResults<P>), &'static str> {
    ops.iter()
        .cloned()
        .try_fold((state, MeasuredResults::new()), fold_modify_state)
}

pub fn get_opfns_and_frontier(q: &Qubit) -> (Vec<&Qubit>, Vec<&StateModifier>) {
    let mut heap = BinaryHeap::new();
    heap.push(q);
    let mut frontier_qubits: Vec<&Qubit> = vec![];
    let mut fn_queue = VecDeque::new();
    while !heap.is_empty() {
        if let Some(q) = heap.pop() {
            match &q.parent {
                Some(parent) => match &parent {
                    Parent::Owned(parents, modifier) => {
                        if let Some(modifier) = modifier {
                            fn_queue.push_front(modifier);
                        }
                        heap.extend(parents);
                    }
                    Parent::Shared(parent) => {
                        let parent = parent.as_ref();
                        if !in_heap(parent, &heap) {
                            heap.push(parent);
                        }
                    }
                },
                None => frontier_qubits.push(q),
            }
            if let Some(deps) = &q.deps {
                deps.iter().for_each(|q| {
                    let q = q.as_ref();
                    if !in_heap(q, &heap) {
                        heap.push(q);
                    }
                })
            }
        }
    }
    (frontier_qubits, fn_queue.into_iter().collect())
}

pub fn get_owned_opfns(q: Qubit) -> Vec<StateModifier> {
    let mut heap = BinaryHeap::new();
    heap.push(q);
    let mut fn_queue = VecDeque::new();
    while !heap.is_empty() {
        if let Some(q) = heap.pop() {
            if let Some(parent) = q.parent {
                match parent {
                    Parent::Owned(parents, modifier) => {
                        if let Some(modifier) = modifier {
                            fn_queue.push_front(modifier);
                        }
                        heap.extend(parents);
                    }
                    Parent::Shared(q) => {
                        if let Ok(q) = Rc::try_unwrap(q) {
                            heap.push(q)
                        }
                    }
                }
            }
            if let Some(deps) = q.deps {
                deps.into_iter().for_each(|q| {
                    if let Ok(q) = Rc::try_unwrap(q) {
                        heap.push(q)
                    }
                })
            }
        }
    }
    fn_queue.into_iter().collect()
}

fn in_heap<T: Eq>(q: T, heap: &BinaryHeap<T>) -> bool {
    for hq in heap {
        if hq == &q {
            return true;
        }
    }
    false
}

/// Create a circuit for the circuit given by `q`. If `natural_order`, then the
/// qubit with index 0 represents the lowest bit in the index of the state (has the smallest
/// increment when flipped), otherwise it's the largest index (which is the internal state used by
/// the simulator).
pub fn make_circuit_matrix<P: Precision>(
    n: u64,
    q: &Qubit,
    natural_order: bool,
) -> Vec<Vec<Complex<P>>> {
    let indices: Vec<u64> = (0..n).collect();
    (0..1 << n)
        .map(|i| {
            let indx = if natural_order {
                i
            } else {
                utils::flip_bits(n as usize, i as u64)
            };
            let (state, _) =
                run_local_with_init(&q, &[(indices.clone(), InitialState::Index(indx))]).unwrap();
            (0..state.state.len())
                .map(|i| {
                    let indx = if natural_order {
                        utils::flip_bits(n as usize, i as u64) as usize
                    } else {
                        i
                    };
                    state.state[indx]
                })
                .collect()
        })
        .collect()
}
