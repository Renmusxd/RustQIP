use super::state_ops::OperatorFn;
use std::rc::Rc;
use crate::qip::qubits::Parent::{Owned, Shared};

#[derive(Debug)]
pub struct Session {
    index: u64
}

impl Session {
    pub fn new() -> Session {
        Session {
            index: 0
        }
    }

    pub fn qubit(&mut self, n: u64) -> Qubit {
        let base_index = self.index;
        self.index = self.index + n;

        Qubit::new((base_index .. self.index).collect())
    }
}

#[derive(Debug)]
pub enum Parent {
    Owned(Qubit),
    Shared(Rc<Qubit>)
}

#[derive(Debug)]
pub struct QubitInheritance {
    pub parents : Vec<Parent>,
    pub operator: OperatorFn
}

#[derive(Debug)]
pub struct Qubit {
    pub indices: Vec<u64>,
    pub parent: Option<QubitInheritance>,
}

impl Qubit {
    fn new(indices: Vec<u64>) -> Qubit {
        Qubit {
            indices,
            parent: None,
        }
    }

    pub fn merge_with_fn(qubits: Vec<Qubit>, operator: OperatorFn) -> Qubit {
        let mut all_indices = Vec::new();

        for q in qubits.iter() {
            all_indices.extend(q.indices.iter());
        }
        all_indices.sort();

        Qubit {
            indices: all_indices,
            parent: Some(QubitInheritance {
                parents: qubits.into_iter().map(|x| Owned(x)).collect(),
                operator
            })
        }
    }

    pub fn merge(qubits: Vec<Qubit>) -> Qubit {
        Qubit::merge_with_fn(qubits, |x| x)
    }

    pub fn split(q: Qubit, selected_indices: Vec<u64>) -> (Qubit, Qubit) {
        let remaining = q.indices.clone()
            .into_iter()
            .filter(|x| selected_indices.contains(x))
            .collect();
        let shared_parent = Rc::new(q);

        (Qubit {
            indices: selected_indices,
            parent: Some(QubitInheritance {
                parents: vec![Shared(shared_parent.clone())],
                operator: |x| x
            })
        }, Qubit {
            indices: remaining,
            parent: Some(QubitInheritance {
                parents: vec![Shared(shared_parent)],
                operator: |x| x
            })
        })
    }
}

