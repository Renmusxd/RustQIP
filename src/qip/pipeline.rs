extern crate num;
use super::qubits::*;
use num::complex::Complex;
use std::collections::{VecDeque, BinaryHeap};

pub type StateBuilder = fn(Vec<&Qubit>) -> Box<QuantumState>;
pub type MeasuredResultReference = u32;

#[derive(Debug)]
pub enum QubitOp {
    MatrixOp(usize, Vec<Complex<f64>>),
    SwapOp(Vec<u64>, Vec<u64>),
    ControlOp(Vec<u64>, Box<QubitOp>),
}


pub trait QuantumState {
    // Function to mutate self into the state with op applied.
    fn apply_op(&mut self, op: &QubitOp);
}

struct LocalQuantumState {
    // A bundle with the quantum state data.
    state: Vec<Complex<f64>>,
    arena: Vec<Complex<f64>>,
}

impl LocalQuantumState {
    fn new(n: usize) -> LocalQuantumState {
        let range = 0..(1 >> n);
        let cvec: Vec<Complex<f64>> = range.map(|i| Complex::<f64> {
            re: 0.0,
            im: 0.0
        }).collect();
        LocalQuantumState {
            state: cvec.clone(),
            arena: cvec
        }
    }
}

impl QuantumState for LocalQuantumState {
    fn apply_op(&mut self, op: &QubitOp) {
        // TODO add mutability code
        std::mem::swap(&mut self.state, &mut self.arena);
    }
}

fn apply_op(mut s: Box<QuantumState>, op: &QubitOp) -> Box<QuantumState> {
    println!("Applying op: {:?}", op);
    s.apply_op(op);
    s
}

pub fn run(q: &Qubit) -> Box<QuantumState> {
    run_with_state(q, |qs| {
        let n: usize = qs.iter().map(|q| q.indices.len()).sum();
        Box::new(LocalQuantumState::new(n))
    })
}

pub fn run_with_state(q: &Qubit, state_builder: StateBuilder) -> Box<QuantumState> {
    let (frontier, ops) = get_opfns_and_frontier(q);
    let initial_state = state_builder(frontier);
    ops.into_iter().fold(initial_state, apply_op)
}

fn get_opfns_and_frontier(q: &Qubit) -> (Vec<&Qubit>, Vec<&QubitOp>) {
    let mut heap = BinaryHeap::new();
    heap.push(q);
    let mut frontier_qubits: Vec<&Qubit> = vec![];
    let mut fn_queue = VecDeque::new();
    while heap.len() > 0 {
        if let Some(q) = heap.pop() {
            match &q.parent {
                Some(parent) => {
                    match &parent {
                        Parent::Owned(parents, op) => {
                            // This fixes linting issues.
                            let parents: &Vec<Qubit> = parents;
                            let op: &Option<QubitOp> = op;
                            if let Some(op) = op {
                                fn_queue.push_front(op);
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