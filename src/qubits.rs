use crate::errors::CircuitError;
use crate::pipeline::*;
use crate::types::Precision;
use crate::Complex;
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

/// Possible relations to a parent Register
#[derive(Debug)]
pub enum Parent {
    /// A set of owned parents by a given Register, that Register has the union of all indices of
    /// its parents. The transition is governed by the StateModifier if present.
    Owned(Vec<Register>, Option<StateModifier>),
    /// A single shared parent, the child has a subsection of the indices of the parent.
    Shared(Rc<Register>),
}

/// A Register object, possible representing multiple physical qubit indices.
pub struct Register {
    /// The set of indices (qubits) represented by this Register.
    pub indices: Vec<u64>,
    /// The parent(s) of this Register (prior in time in the quantum circuit).
    pub parent: Option<Parent>,
    /// Additional dependencies for this Register (such as if it relies on the classical
    /// measurements of other Register).
    pub deps: Option<Vec<Rc<Register>>>,
    /// The unique ID of this Register.
    pub id: u64,
}

impl Register {
    pub(crate) fn new(id: u64, indices: Vec<u64>) -> Result<Register, CircuitError> {
        if indices.is_empty() {
            CircuitError::make_str_err("Register must have at least one index assigned.")
        } else {
            Ok(Register {
                indices,
                parent: None,
                deps: None,
                id,
            })
        }
    }

    /// Create a handle for feeding values.
    pub fn handle(&self) -> RegisterHandle {
        RegisterHandle {
            indices: self.indices.clone(),
        }
    }

    /// Merge Registers to form a new Register object.
    pub fn merge_with_modifier(
        id: u64,
        registers: Vec<Register>,
        modifier: Option<StateModifier>,
    ) -> Register {
        let all_indices = registers
            .iter()
            .map(|r| r.indices.clone())
            .flatten()
            .collect();

        Register {
            indices: all_indices,
            parent: Some(Parent::Owned(registers, modifier)),
            deps: None,
            id,
        }
    }

    /// Split the relative indices out of `r` into its own qubit, remaining live in second qubit.
    pub fn split(
        ida: u64,
        idb: u64,
        r: Register,
        indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError> {
        for indx in &indices {
            if *indx > r.n() {
                let message = format!("All indices for splitting must be below r.n = {:?}", r.n());
                return CircuitError::make_err(message);
            }
        }
        if indices.len() == r.indices.len() {
            CircuitError::make_str_err("Cannot split out all indices into own Registers.")
        } else if indices.is_empty() {
            CircuitError::make_str_err("Must provide indices to split.")
        } else {
            let selected_indices: Vec<u64> =
                indices.into_iter().map(|i| r.indices[i as usize]).collect();
            Self::split_absolute(ida, idb, r, selected_indices)
        }
    }

    /// Split a Register in two, with one having the indices in `selected_indices`
    pub fn split_absolute(
        ida: u64,
        idb: u64,
        r: Register,
        selected_indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError> {
        if selected_indices.len() == r.indices.len() {
            return CircuitError::make_str_err("Cannot split out all indices into own Registers.");
        } else if selected_indices.is_empty() {
            return CircuitError::make_str_err("Must provide indices to split.");
        }
        for indx in &selected_indices {
            if !r.indices.contains(indx) {
                let message = format!(
                    "Index {:?} not found in Register with indices {:?}",
                    indx, r.indices
                );
                return CircuitError::make_err(message);
            }
        }

        let remaining = r
            .indices
            .clone()
            .into_iter()
            .filter(|x| !selected_indices.contains(x))
            .collect();
        let shared_parent = Rc::new(r);

        Ok((
            Register {
                indices: selected_indices,
                parent: Some(Parent::Shared(shared_parent.clone())),
                deps: None,
                id: ida,
            },
            Register {
                indices: remaining,
                parent: Some(Parent::Shared(shared_parent.clone())),
                deps: None,
                id: idb,
            },
        ))
    }

    /// Make a measurement handle and a Register which depends on that measurement.
    pub fn make_measurement_handle(id: u64, r: Register) -> (Register, MeasurementHandle) {
        let indices = r.indices.clone();
        let shared_parent = Rc::new(r);
        let handle = MeasurementHandle::new(&shared_parent);
        (
            Register {
                indices,
                parent: Some(Parent::Shared(shared_parent)),
                deps: None,
                id,
            },
            handle,
        )
    }

    /// Add additional Register dependencies to a given Register.
    pub fn add_deps(r: Register, deps: Vec<Rc<Register>>) -> Register {
        Register {
            indices: r.indices,
            parent: r.parent,
            deps: Some(deps),
            id: r.id,
        }
    }

    /// Get number of qubits in this Register object
    pub fn n(&self) -> u64 {
        self.indices.len() as u64
    }
}

impl Eq for Register {}

impl PartialEq for Register {
    fn eq(&self, other: &Register) -> bool {
        self.id == other.id
    }
}

impl Ord for Register {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Register {
    fn partial_cmp(&self, other: &Register) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let int_strings = self
            .indices
            .iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(
            f,
            "Register[{}][{}]",
            self.id.to_string(),
            int_strings.join(", ")
        )
    }
}

/// A Register handle for using when setting initial states for the circuit.
#[derive(Debug)]
pub struct RegisterHandle {
    indices: Vec<u64>,
}

impl RegisterHandle {
    /// Make an initial state for the handle using an index: `|index>`
    pub fn make_init_from_index<P: Precision>(
        &self,
        index: u64,
    ) -> Result<RegisterInitialState<P>, CircuitError> {
        let n = self.indices.len();
        if index < (1 << n) as u64 {
            Ok((self.indices.clone(), InitialState::Index(index)))
        } else {
            let message = format!(
                "Index {:?} is too large, must be less than 2^{:?}",
                index, n
            );
            CircuitError::make_err(message)
        }
    }

    /// Make an initial state for the handle given a fully qualified state: `a|0> + b|1> + c|2> ...`
    pub fn make_init_from_state<P: Precision>(
        &self,
        state: Vec<Complex<P>>,
    ) -> Result<RegisterInitialState<P>, CircuitError> {
        let n = self.indices.len();
        if state.len() == 1 << n {
            Ok((self.indices.clone(), InitialState::FullState(state)))
        } else {
            let message = format!(
                "State not correct size for RegisterHandle (is {}, must be 2^{})",
                state.len(),
                n
            );
            CircuitError::make_err(message)
        }
    }
}
