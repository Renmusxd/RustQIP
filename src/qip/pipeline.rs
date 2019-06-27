extern crate num;

use std::collections::{BinaryHeap, VecDeque};

use num::complex::Complex;
use std::collections::HashMap;
use std::cmp;
use crate::qubits::*;
use crate::state_ops::*;
use crate::measurement_ops::measure;
use crate::utils;
use std::cmp::max;

pub enum StateModifierType {
    UnitaryOp(QubitOp),
    MeasureState(u64, Vec<u64>)
}

pub struct StateModifier {
    name: String,
    modifier: StateModifierType
}

impl StateModifier {
    pub fn new_unitary(name: String, op: QubitOp) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::UnitaryOp(op)
        }
    }

    pub fn new_measurement(name: String, id: u64, indices: Vec<u64>) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::MeasureState(id, indices)
        }
    }
}

pub struct MeasuredResults {
    pub results: HashMap<u64, (u64, f64)>
}

impl MeasuredResults {
    pub fn new() -> MeasuredResults {
        MeasuredResults {
            results: HashMap::new()
        }
    }
}

/// A trait which represents the state of the qubits
pub trait QuantumState {
    /// Make new state with n qubits
    fn new(n: u64) -> Self;

    /// Initialize new state with initial states.
    fn new_from_initial_states(n: u64, states: &[QubitInitialState]) -> Self;

    /// Function to mutate self into the state with op applied.
    fn apply_op(&mut self, op: &QubitOp);

    /// Mutate self with measurement, return result as index and probability
    fn measure(&mut self, indices: &Vec<u64>) -> (u64, f64);
}

/// A basic representation of a quantum state, given by a vector of complex numbers stored
/// locally on the machine (plus an arena of equal size to work in).
pub struct LocalQuantumState {
    // A bundle with the quantum state data.
    pub n: u64,
    pub state: Vec<Complex<f64>>,
    arena: Vec<Complex<f64>>,
}

pub enum InitialState {
    FullState(Vec<Complex<f64>>),
    Index(u64)
}

pub type QubitInitialState = (Vec<u64>, InitialState);

impl QuantumState for LocalQuantumState {
    /// Build a new LocalQuantumState
    fn new(n: u64) -> LocalQuantumState {
        LocalQuantumState::new_from_initial_states(n, &vec![])
    }

    /// Build a local state using a set of initial states for subsets of the qubits.
    /// These initial states are made from the qubit handles.
    fn new_from_initial_states(n: u64, states: &[QubitInitialState]) -> LocalQuantumState {
        let max_init_n = states.iter().map(|(indices, _)| indices).cloned().flatten().max().map(|m| m+1);

        let n = max_init_n.map(|m| max(n, m)).unwrap_or(n);

        let mut cvec: Vec<Complex<f64>> = (0.. 1 << n).map(|_| Complex::<f64> {
            re: 0.0,
            im: 0.0,
        }).collect();

        // Assume that all unrepresented indices are in the |0> state.
        let n_fullindices: u64 = states.iter().map(|(indices, state)| {
            match state {
                InitialState::FullState(_) => indices.len() as u64,
                _ => 0
            }
        }).sum();

        // Make the index template/base
        let template: u64 = states.iter().map(|(indices, state)| -> u64 {
            match state {
                InitialState::Index(val_indx) => {
                    indices.iter().enumerate().map(|(i, indx)| {
                        let bit = (val_indx >> i) & 1;
                        bit << (n - 1 - indx)
                    }).sum()
                }
                _ => 0
            }
        }).sum();

        let init = Complex::<f64> {
            re: 1.0,
            im: 0.0
        };
        // Go through each combination of full index locations
        (0 .. 1 << n_fullindices).for_each(|i| {
            // Calculate the offset from template, and the product of fullstates.
            let (delta_index, _, val) = states.iter().fold((0u64, 0u64, init), |acc, (indices, state) | {
                if let InitialState::FullState(vals) = state {
                    let (superindex_acc, sub_index_offset, val_acc) = acc;
                    // Now we need to make additions to the superindex by adding bits based on
                    // indices, as well as return the value given by the [sub .. sub + len] bits
                    // from i.
                    let index_mask = (1 << indices.len() as u64) - 1;
                    let val_index_bits = (i >> sub_index_offset) & index_mask;
                    let val_acc = val_acc * vals[val_index_bits as usize];

                    let superindex_delta: u64 = indices.iter().enumerate().map(|(j,indx)| {
                        let bit = (val_index_bits >> j as u64) & 1u64;
                        bit << (n - 1 - indx)
                    }).sum();
                    (superindex_acc + superindex_delta, sub_index_offset + indices.len() as u64, val_acc)
                } else {
                    acc
                }
            });
            cvec[(delta_index + template) as usize] = val;
        });

        LocalQuantumState {
            n,
            state: cvec.clone(),
            arena: cvec,
        }
    }

    fn apply_op(&mut self, op: &QubitOp) {
        apply_op(self.n, op, &self.state, &mut self.arena, 0, 0, self.n > PARALLEL_THRESHOLD);
        std::mem::swap(&mut self.state, &mut self.arena);
    }

    fn measure(&mut self, indices: &Vec<u64>) -> (u64, f64) {
        let measured_result = measure(self.n, indices, &self.state, &mut self.arena, 0, 0);
        std::mem::swap(&mut self.state, &mut self.arena);
        measured_result
    }
}

/// Apply an QubitOp to the state `s` and return the new state.
fn fold_modify_state<QS: QuantumState>(mut acc: (QS, MeasuredResults), modifier: &StateModifier) -> (QS, MeasuredResults) {
    let (mut s, mut mr) = acc;
    match &modifier.modifier {
        StateModifierType::UnitaryOp(op) => s.apply_op(op),
        StateModifierType::MeasureState(id, indices) => {
            let result = s.measure(indices);
            mr.results.insert(id.clone(), result);
        }
    }
    (s, mr)
}


/// Builds a default state of size `n`
pub fn run<QS: QuantumState>(q: &Qubit) -> (QS, MeasuredResults) {
    run_with_statebuilder(q, |qs| -> QS {
        let n: u64 = qs.iter().map(|q| q.indices.len() as u64).sum();
        QS::new(n)
    })
}

pub fn run_with_init<QS: QuantumState>(q: &Qubit, states: &[QubitInitialState]) -> (QS, MeasuredResults){
    run_with_statebuilder(q, |qs| -> QS {
        let n: u64 = qs.iter().map(|q| q.indices.len() as u64).sum();
        QS::new_from_initial_states(n, states)
    })
}

pub fn run_with_statebuilder<QS: QuantumState, F: FnOnce(Vec<&Qubit>) -> QS>(q: &Qubit, state_builder: F) -> (QS, MeasuredResults) {
    let (frontier, ops) = get_opfns_and_frontier(q);
    let state = state_builder(frontier);
    ops.into_iter().fold((state, MeasuredResults::new()), fold_modify_state)
}

/// `run` the pipeline using `LocalQuantumState`.
pub fn run_local(q: &Qubit) -> (LocalQuantumState, MeasuredResults) {
    run(q)
}

/// `run_with_init` the pipeline using `LocalQuantumState`
pub fn run_local_with_init(q: &Qubit, states: &[QubitInitialState]) -> (LocalQuantumState, MeasuredResults) {
    run_with_init(q, states)
}

fn get_opfns_and_frontier(q: &Qubit) -> (Vec<&Qubit>, Vec<&StateModifier>) {
    let mut heap = BinaryHeap::new();
    heap.push(q);
    let mut frontier_qubits: Vec<&Qubit> = vec![];
    let mut fn_queue = VecDeque::new();
    while heap.len() > 0 {
        if let Some(q) = heap.pop() {
            match &q.parent {
                Some(parent) => {
                    match &parent {
                        Parent::Owned(parents, modifier) => {
                            // This fixes linting issues.
                            let parents: &Vec<Qubit> = parents;
                            let modifier: &Option<StateModifier> = modifier;
                            if let Some(modifier) = modifier {
                                fn_queue.push_front(modifier);
                            }
                            heap.extend(parents);
                        }
                        Parent::Shared(parent) => {
                            let parent = parent.as_ref();
                            if !qubit_in_heap(parent, &heap) {
                                heap.push(parent);
                            }
                        }
                    }
                }
                None => frontier_qubits.push(q)
            }
        }
    }
    (frontier_qubits, fn_queue.into_iter().collect())
}

fn qubit_in_heap(q: &Qubit, heap: &BinaryHeap<&Qubit>) -> bool {
    for hq in heap {
        if hq == &q {
            return true;
        }
    }
    false
}

pub fn make_circuit_matrix(n: u64, q: &Qubit, h: QubitHandle) -> Vec<Vec<Complex<f64>>> {
    (0..1 << n).map(|i| {
        let (state, _) = run_local_with_init(&q, &[
            h.make_init_from_index(i).unwrap()
        ]);
        (0 .. state.state.len()).map(|i| {
            state.state[utils::flip_bits(n as usize, i as u64) as usize]
        }).collect()
    }).collect()
}