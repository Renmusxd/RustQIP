use super::pipeline::OperatorFn;
use std::rc::Rc;
use crate::qip::qubits::Parent::{Owned, Shared};
use std::fmt;


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

pub enum Parent {
    Owned(Vec<Qubit>, Option<OperatorFn>),
    Shared(Rc<Qubit>)
}

pub struct QubitInheritance {
    pub parent : Parent
}

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

    pub fn merge_with_fn(qubits: Vec<Qubit>, operator: Option<OperatorFn>) -> Qubit {
        let mut all_indices = Vec::new();

        for q in qubits.iter() {
            all_indices.extend(q.indices.iter());
        }
        all_indices.sort();

        Qubit {
            indices: all_indices,
            parent: Some(QubitInheritance {
                parent: Owned(qubits, operator)
            })
        }
    }

    pub fn merge(qubits: Vec<Qubit>) -> Qubit {
        Qubit::merge_with_fn(qubits, None)
    }

    pub fn split(q: Qubit, selected_indices: Vec<u64>) -> (Qubit, Qubit) {
        let remaining = q.indices.clone()
            .into_iter()
            .filter(|x| !selected_indices.contains(x))
            .collect();
        let shared_parent = Rc::new(q);

        (Qubit {
            indices: selected_indices,
            parent: Some(QubitInheritance {
                parent: Shared(shared_parent.clone())
            })
        }, Qubit {
            indices: remaining,
            parent: Some(QubitInheritance {
                parent: Shared(shared_parent.clone())
            })
        })
    }
}


impl fmt::Debug for Qubit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let int_strings = self.indices.iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "Qubit[{}]", int_strings.join(", "))
    }
}