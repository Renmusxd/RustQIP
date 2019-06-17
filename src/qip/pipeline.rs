extern crate num;

use std::collections::{BinaryHeap, VecDeque};

use num::complex::Complex;
use std::collections::HashMap;

use super::qubits::*;
use super::state_ops::*;
use crate::measurement_ops::measure;

pub type StateBuilder<QS: QuantumState> = fn(Vec<&Qubit>) -> QS;


pub enum StateModifier {
    UnitaryOp(QubitOp),
    MeasureState(u64, Vec<u64>)
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
    /// Function to mutate self into the state with op applied.
    fn apply_op(&mut self, op: &QubitOp);

    /// Mutate self with measurement, return result as index and probability
    fn measure(&mut self, indices: &Vec<u64>) -> (u64, f64);
}

/// A basic representation of a quantum state, given by a vector of complex numbers stored
/// locally on the machine (plus an arena of equal size to work in).
pub struct LocalQuantumState {
    // A bundle with the quantum state data.
    n: u64,
    state: Vec<Complex<f64>>,
    arena: Vec<Complex<f64>>,
}

pub enum InitialState {
    FullState(Vec<Complex<f64>>),
    Index(u64)
}

impl LocalQuantumState {
    /// Build a new LocalQuantumState
    fn new(n: u64) -> LocalQuantumState {
        LocalQuantumState::new_from_initial_states(n, vec![])
    }

    /// Build a local state using a set of initial states for subsets of the qubits.
    ///
    /// # Example
    /// ```
    /// // TODO fill in example
    /// ```
    pub fn new_from_initial_states(n: u64, states: Vec<(Vec<u64>, InitialState)>) -> LocalQuantumState {
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
                InitialState::Index(indx) => {
                    indices.iter().enumerate().map(|(i, indx)| {
                        let bit = (indx >> i) & 1;
                        bit << indx
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
                        bit << indx
                    }).sum();

                    (superindex_acc + superindex_delta, sub_index_offset + indices.len() as u64, val_acc)
                } else {
                    acc
                }
            });

        });

        LocalQuantumState {
            n,
            state: cvec.clone(),
            arena: cvec,
        }
    }
}

impl QuantumState for LocalQuantumState {
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
    match modifier {
        StateModifier::UnitaryOp(op) => s.apply_op(op),
        StateModifier::MeasureState(id, indices) => {
            let result = s.measure(indices);
            mr.results.insert(id.clone(), result);
        }
    }
    (s, mr)
}

/// Run the pipeline using `LocalQuantumState` to store state.
pub fn run(q: &Qubit) -> (LocalQuantumState, MeasuredResults) {
    run_with_state(q, |qs| {
        let n: u64 = qs.iter().map(|q| q.indices.len() as u64).sum();
        LocalQuantumState::new(n)
    })
}

pub fn run_with_state<QS: QuantumState>(q: &Qubit, state_builder: StateBuilder<QS>) -> (QS, MeasuredResults) {
    let (frontier, ops) = get_opfns_and_frontier(q);
    let state = state_builder(frontier);
    ops.into_iter().fold((state, MeasuredResults::new()), fold_modify_state)
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
                            heap.extend(parents.iter());
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