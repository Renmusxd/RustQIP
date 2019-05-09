
#[derive(Debug)]
pub struct Session {
    index: i32
}

impl Session {
    pub fn new() -> Session {
        Session {
            index: 0
        }
    }

    pub fn qubit(&mut self, n: i32) -> Qubit {
        let base_index = self.index;
        self.index = self.index + n;

        Qubit::new((base_index .. self.index).collect())
    }
}


pub struct QuantumState {
  // A bundle with the quantum state data.
}


type OperatorFn = fn(QuantumState) -> QuantumState;

#[derive(Debug)]
pub struct QubitInheritance {
    pub parents : Vec<Qubit>,
    pub operator: OperatorFn
}

#[derive(Debug)]
pub struct Qubit {
    pub indices: Vec<i32>,
    pub parent: Option<QubitInheritance>,
}

impl Qubit {
    fn new(indices: Vec<i32>) -> Qubit {
        Qubit {
            indices,
            parent: None,
        }
    }

    pub fn merge(qubits: Vec<Qubit>) -> Qubit {
        let mut all_indices = Vec::new();

        for q in qubits.iter() {
            all_indices.extend(q.indices.iter());
        }
        all_indices.sort();

        Qubit {
            indices: all_indices,
            parent: Some(QubitInheritance {
                parents: qubits,
                operator: |x| x
            })
        }
    }
}

