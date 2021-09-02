use std::cmp::{max, Ordering};
use std::collections::HashMap;
use std::collections::{BinaryHeap, VecDeque};

use crate::errors::CircuitError;
use crate::measurement_ops::{
    measure, measure_prob, measure_probs, prob_magnitude, soft_measure, MeasuredCondition,
};
use crate::qubits::Parent;
use crate::rayon_helper::*;
use crate::state_ops::*;
use crate::utils::flip_bits;
use crate::*;
use num::{One, Zero};
use std::fmt;
use std::rc::Rc;

/// A functions which maps measured values to a series of StateModifiers which will be applied to
/// the state.
pub type SideChannelModifierFn = dyn Fn(&[u64]) -> Result<Vec<StateModifier>, CircuitError>;

/// The set of ways to modify a QuantumState
pub enum StateModifierType {
    /// Ops such as matrices, swaps, and conditions
    UnitaryOp(UnitaryOp),
    /// Measurements of the quantum state
    MeasureState(u64, Vec<u64>, f64),
    /// Stochastic measurements which don't affect the state.
    StochasticMeasureState(u64, Vec<u64>, f64),
    /// Subsections of the circuit which depend on measured values.
    SideChannelModifiers(Vec<MeasurementHandle>, Box<SideChannelModifierFn>),
    /// Debugging op
    Debug(Vec<Vec<u64>>, Box<dyn Fn(Vec<Vec<f64>>)>),
}

impl fmt::Debug for StateModifierType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let to_strs = |indices: &Vec<u64>| {
            indices
                .iter()
                .map(|x| x.clone().to_string())
                .collect::<Vec<String>>()
        };

        match self {
            StateModifierType::UnitaryOp(op) => write!(f, "UnitaryOp[{:?}]", op),
            StateModifierType::MeasureState(id, indices, angle) => write!(
                f,
                "MeasureState[{:?}, {:?}, {:?}]",
                id,
                to_strs(indices),
                angle
            ),
            StateModifierType::StochasticMeasureState(id, indices, angle) => write!(
                f,
                "StochasticMeasureState[{:?}, {:?}, {:?}]",
                id,
                to_strs(indices),
                angle
            ),
            StateModifierType::SideChannelModifiers(handle, _) => {
                write!(f, "SideChannelModifiers[{:?}]", handle)
            }
            StateModifierType::Debug(indices, _) => write!(f, "Debug[{:?}]", indices),
        }
    }
}

/// A named state modifier.
#[derive(Debug)]
pub struct StateModifier {
    /// Name of modifier.
    pub name: String,
    /// Mechanism of modifier.
    pub modifier: StateModifierType,
}

impl StateModifier {
    /// Create a new unitary state modifier (matrices, swaps, ...)
    pub fn new_unitary(name: String, op: UnitaryOp) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::UnitaryOp(op),
        }
    }

    /// Create a new measurement state modifier.
    pub fn new_measurement(name: String, id: u64, indices: Vec<u64>) -> StateModifier {
        StateModifier::new_measurement_basis(name, id, indices, 0.0)
    }

    /// Create a new measurement state modifier on an off-computational basis:
    /// `cos(angle)|0> + sin(angle)|1>`
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

    /// Create a new stochastic measurement which doesn't affect the state but gives the adds
    /// the chance of each state.
    pub fn new_stochastic_measurement(name: String, id: u64, indices: Vec<u64>) -> StateModifier {
        StateModifier::new_stochastic_measurement_basis(name, id, indices, 0.0)
    }

    /// Create a new stochastic measurement state modifier on an off-computational basis:
    /// `cos(angle)|0> + sin(angle)|1>`
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

    /// Create a new side channel state modifier which builds part of the circuit dependent on
    /// the measured values from previous steps.
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

    /// Create a new debug state modifier (which doesn't modify the state).
    pub fn new_debug(
        name: String,
        indices: Vec<Vec<u64>>,
        f: Box<dyn Fn(Vec<Vec<f64>>)>,
    ) -> StateModifier {
        StateModifier {
            name,
            modifier: StateModifierType::Debug(indices, f),
        }
    }
}

/// A handle which can be used to retrieve measured values.
#[derive(Debug)]
pub struct MeasurementHandle {
    register: Rc<Register>,
}

impl MeasurementHandle {
    /// Build a new MeasurementHandle from a Register being measured.
    pub fn new(register: &Rc<Register>) -> Self {
        MeasurementHandle {
            register: register.clone(),
        }
    }

    /// Get a cloned reference of the measured Register.
    pub fn clone_register(&self) -> Rc<Register> {
        self.register.clone()
    }

    /// Get the id of this MeasurementHandle
    pub fn get_id(&self) -> u64 {
        self.register.id
    }
}

impl Clone for MeasurementHandle {
    fn clone(&self) -> Self {
        MeasurementHandle {
            register: self.register.clone(),
        }
    }
}

impl Eq for MeasurementHandle {}

impl PartialEq for MeasurementHandle {
    fn eq(&self, other: &MeasurementHandle) -> bool {
        self.register == other.register
    }
}

impl Ord for MeasurementHandle {
    fn cmp(&self, other: &Self) -> Ordering {
        self.register.cmp(&other.register)
    }
}

impl PartialOrd for MeasurementHandle {
    fn partial_cmp(&self, other: &MeasurementHandle) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A struct which provides the measured values from the circuit.
#[derive(Default, Debug)]
pub struct MeasuredResults<P: Precision> {
    results: HashMap<u64, (u64, P)>,
    stochastic_results: HashMap<u64, Vec<P>>,
}

impl<P: Precision> MeasuredResults<P> {
    /// Make a new MeasuredResults container.
    pub fn new() -> MeasuredResults<P> {
        MeasuredResults::default()
    }

    /// Retrieve the measurement and likelihood for a given handle.
    pub fn get_measurement(&self, handle: &MeasurementHandle) -> Option<(u64, P)> {
        self.results.get(&handle.get_id()).cloned()
    }

    /// Retrieve the set of measurements for a given handle.
    pub fn get_stochastic_measurements(&self, handle: u64) -> Option<&[P]> {
        self.stochastic_results.get(&handle).map(|x| x.as_slice())
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

/// Order of qubits returned by `QuantumState::into_state` and other similar methods.
#[derive(Clone, Copy, Debug, Hash, PartialEq)]
pub enum Representation {
    /// Qubit with index 0 is the least significant index bit.
    LittleEndian,
    /// Qubit with index 0 is the most significant index bit.
    BigEndian,
}

/// A trait which represents the state of the qubits
pub trait QuantumState<P: Precision> {
    /// Make new state with n qubits
    fn new(n: u64) -> Self;

    /// Initialize new state with initial states.
    fn new_from_initial_states(n: u64, states: &[RegisterInitialState<P>]) -> Self;

    /// Get number of qubits represented by this state.
    fn n(&self) -> u64;

    /// Function to mutate self into the state with op applied.
    fn apply_op(&mut self, op: &UnitaryOp) {
        self.apply_op_with_name(None, op)
    }

    /// Apply op with a given name. Mutate self using op.
    fn apply_op_with_name(&mut self, name: Option<&str>, op: &UnitaryOp);

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
    fn into_state(self, order: Representation) -> Vec<Complex<P>>;

    /// Rotate to a new computational basis:
    /// `|0'> =  cos(angle)|0> + sin(angle)|1>`
    /// `|1'> = -sin(angle)|0> + cos(angle)|1>`
    fn rotate_basis(&mut self, indices: &[u64], angle: f64) {
        if angle != 0.0 {
            let (sangle, cangle) = angle.sin_cos();
            let basis_mat = from_reals(&[cangle, -sangle, sangle, cangle]);
            indices.iter().for_each(|indx| {
                let op = make_matrix_op(vec![*indx], basis_mat.clone()).unwrap();
                self.apply_op(&op);
            });
        }
    }
}

/// A basic representation of a quantum state, given by a vector of complex numbers stored
/// locally on the machine (plus an arena of equal size to work in).
#[derive(Debug)]
pub struct LocalQuantumState<P: Precision> {
    // A bundle with the quantum state data.
    n: u64,
    state: Vec<Complex<P>>,
    arena: Vec<Complex<P>>,
}

/// Fill a buffer from the initial states.
pub fn fill_buffer_initial_state<P: Precision>(
    n: u64,
    states: &[RegisterInitialState<P>],
    buffer: &mut [Complex<P>],
) {
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
            InitialState::Index(val_indx) => {
                let val_indx = flip_bits(indices.len(), *val_indx);
                sub_to_full(n, indices, val_indx, acc)
            }
            _ => acc,
        }
    });

    // Go through each combination of full index locations
    (0..1 << n_fullindices).for_each(|i| {
        // Calculate the offset from template, and the product of fullstates.
        let (delta_index, val) = create_state_entry(n, i, states);
        buffer[(delta_index + template) as usize] = val;
    });
}

impl<P: Precision> LocalQuantumState<P> {
    /// Build a local state using a set of initial states for subsets of the qubits.
    /// These initial states are made from the Register handles.
    fn new_from_initial_states_and_multithread(
        n: u64,
        states: &[RegisterInitialState<P>],
    ) -> LocalQuantumState<P> {
        let max_init_n = states
            .iter()
            .map(|(indices, _)| indices)
            .cloned()
            .flatten()
            .max()
            .map(|m| m + 1);

        let n = max_init_n.map_or(n, |m| max(n, m));

        let mut cvec: Vec<Complex<P>> = (0..1 << n).map(|_| Complex::default()).collect();
        fill_buffer_initial_state(n, states, &mut cvec);

        let arena = vec![Complex::zero(); cvec.len()];
        LocalQuantumState {
            n,
            state: cvec,
            arena,
        }
    }

    /// Make a new LocalQuantumState from a fully defined state.
    pub fn new_from_full_state(
        n: u64,
        state: Vec<Complex<P>>,
        order: Representation,
    ) -> Result<LocalQuantumState<P>, CircuitError> {
        if state.len() != 1 << n as usize {
            let message = format!(
                "Provided state is not the correct size, expected {:?} but found {:?}",
                1 << n,
                state.len()
            );
            return CircuitError::make_err(message);
        }

        let arena = vec![Complex::zero(); state.len()];

        let state = match order {
            Representation::LittleEndian => {
                let mut state: Vec<_> = state.into_iter().enumerate().collect();
                state.sort_by_key(|(indx, _)| flip_bits(n as usize, *indx as u64));
                state.into_iter().map(|(_, c)| c).collect()
            }
            Representation::BigEndian => state,
        };

        Ok(LocalQuantumState { n, state, arena })
    }

    /// Return a reference to the internal state.
    pub fn state_ref(&self) -> &Vec<Complex<P>> {
        &self.state
    }

    /// Return a mutable reference to the internal state.
    pub fn mut_state_ref(&mut self) -> &mut Vec<Complex<P>> {
        &mut self.state
    }

    /// Clone the state in either the `natural_order` or the internal order.
    pub fn clone_state(&mut self, order: Representation) -> Vec<Complex<P>> {
        match order {
            Representation::LittleEndian => {
                let n = self.n;
                let state = &self.state;
                let f = |(i, outputloc): (usize, &mut Complex<P>)| {
                    *outputloc = state[flip_bits(n as usize, i as u64) as usize];
                };

                iter_mut!(self.arena).enumerate().for_each(f);
                self.arena.clone()
            }
            Representation::BigEndian => self.state.clone(),
        }
    }
}

impl<P: Precision> Clone for LocalQuantumState<P> {
    fn clone(&self) -> Self {
        LocalQuantumState {
            n: self.n,
            state: self.state.clone(),
            arena: self.arena.clone(),
        }
    }
}

/// An initial state supplier for building quantum states.
#[derive(Debug, Clone)]
pub enum InitialState<P: Precision> {
    /// A fully qualified state, each |x> has an amplitude
    FullState(Vec<Complex<P>>),
    /// A single index with the whole weight.
    Index(u64),
}

impl<P: Precision> InitialState<P> {
    pub(crate) fn get_amplitude(&self, m: u64) -> Complex<P> {
        match &self {
            InitialState::FullState(v) => v[m as usize],
            InitialState::Index(index) => {
                if *index == m {
                    Complex::one()
                } else {
                    Complex::zero()
                }
            }
        }
    }

    pub(crate) fn get_magnitude(&self) -> P {
        match &self {
            InitialState::Index(_) => P::one(),
            InitialState::FullState(v) => v.iter().map(|c| c.norm_sqr()).sum::<P>().sqrt(),
        }
    }
}

/// A set of indices and their initial state.
pub type RegisterInitialState<P> = (Vec<u64>, InitialState<P>);

impl<P: Precision> QuantumState<P> for LocalQuantumState<P> {
    /// Build a new LocalQuantumState
    fn new(n: u64) -> LocalQuantumState<P> {
        LocalQuantumState::new_from_initial_states(n, &[])
    }

    /// Build a local state using a set of initial states for subsets of the qubits.
    /// These initial states are made from the qubit handles.
    fn new_from_initial_states(n: u64, states: &[RegisterInitialState<P>]) -> LocalQuantumState<P> {
        Self::new_from_initial_states_and_multithread(n, states)
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn apply_op_with_name(&mut self, _name: Option<&str>, op: &UnitaryOp) {
        apply_op(self.n, op, &self.state, &mut self.arena, 0, 0);
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
            soft_measure(self.n, indices, &self.state, None)
        };
        let p = measure_prob(self.n, m, indices, &self.state, None);
        self.rotate_basis(indices, -angle);
        (m, p)
    }

    fn state_magnitude(&self) -> P {
        prob_magnitude(&self.state)
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        self.rotate_basis(indices, angle);
        let probs = measure_probs(self.n, indices, &self.state, None);
        self.rotate_basis(indices, -angle);
        probs
    }

    fn into_state(mut self, order: Representation) -> Vec<Complex<P>> {
        match order {
            Representation::LittleEndian => {
                let n = self.n;
                let state = self.state;
                let f = |(i, outputloc): (usize, &mut Complex<P>)| {
                    *outputloc = state[flip_bits(n as usize, i as u64) as usize];
                };

                iter_mut!(self.arena).enumerate().for_each(f);
                self.arena
            }
            Representation::BigEndian => self.state,
        }
    }
}

pub(crate) fn create_state_entry<P: Precision>(
    n: u64,
    i: u64,
    states: &[RegisterInitialState<P>],
) -> (u64, Complex<P>) {
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
    (delta_index, val)
}

/// Apply an QubitOp to the state `s` and return the new state.
fn fold_modify_state<P: Precision, QS: QuantumState<P>>(
    acc: (QS, MeasuredResults<P>),
    modifier: &StateModifier,
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    let (mut s, mut mr) = acc;
    match &modifier.modifier {
        StateModifierType::UnitaryOp(op) => {
            s.apply_op_with_name(Some(&modifier.name), op);
            Ok((s, mr))
        }
        StateModifierType::MeasureState(id, indices, angle) => {
            let result = s.measure(indices, None, *angle);
            mr.results.insert(*id, result);
            Ok((s, mr))
        }
        StateModifierType::StochasticMeasureState(id, indices, angle) => {
            let result = s.stochastic_measure(indices, *angle);
            mr.stochastic_results.insert(*id, result);
            Ok((s, mr))
        }
        StateModifierType::SideChannelModifiers(handles, f) => {
            let measured_values: Vec<_> = handles
                .iter()
                .map(|handle| mr.get_measurement(handle))
                .collect();
            measured_values.iter().try_for_each(|x| match x {
                Some(_) => Ok(()),
                None => CircuitError::make_str_err("Not all measurements found"),
            })?;
            let measured_values: Vec<_> = measured_values
                .into_iter()
                .map(|m| m.map(|(m, _)| m).unwrap())
                .collect();
            let modifiers = f(&measured_values)?;
            modifiers.iter().try_fold((s, mr), fold_modify_state)
        }
        StateModifierType::Debug(index_groups, f) => {
            let result = index_groups
                .iter()
                .map(|indices| s.stochastic_measure(indices, 0.0))
                .map(|vp| vp.into_iter().map(|p| p.to_f64().unwrap()).collect())
                .collect();
            f(result);
            Ok((s, mr))
        }
    }
}

/// Return the required number of qubits for a given frontier of qubits (those in the circuit
/// with no parent qubits).
pub fn get_required_state_size_from_frontier(frontier: &[&Register]) -> u64 {
    frontier
        .iter()
        .map(|r| &r.indices)
        .cloned()
        .flatten()
        .max()
        .map_or(0, |m| m + 1)
}

/// Return the required number of qubits for a given frontier of Registers (those in the circuit
/// with no parent Registers) and a set of initial states.
pub fn get_required_state_size<P: Precision>(
    frontier: &[&Register],
    states: &[RegisterInitialState<P>],
) -> u64 {
    let max_qubit_n = get_required_state_size_from_frontier(frontier);
    let max_init_n = states
        .iter()
        .map(|(indices, _)| indices)
        .cloned()
        .flatten()
        .max()
        .map_or(0, |m| m + 1);
    max(max_init_n, max_qubit_n)
}

/// Builds a default state of size `n`
pub fn run<P: Precision, QS: QuantumState<P>>(
    r: &Register,
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    run_with_statebuilder(r, |rs| -> Result<QS, CircuitError> {
        let n = get_required_state_size_from_frontier(&rs);
        Ok(QS::new(n))
    })
}

/// Run the circuit, starting by building the quantum state QS with a set of initial states.
pub fn run_with_init<P: Precision, QS: QuantumState<P>>(
    r: &Register,
    states: &[RegisterInitialState<P>],
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    run_with_statebuilder(r, |rs| -> Result<QS, CircuitError> {
        let n = get_required_state_size(&rs, states);
        Ok(QS::new_from_initial_states(n, states))
    })
}

/// Run the circuit using a function to build the initial state.
pub fn run_with_statebuilder<
    P: Precision,
    QS: QuantumState<P>,
    F: FnOnce(Vec<&Register>) -> Result<QS, CircuitError>,
>(
    r: &Register,
    state_builder: F,
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    let (frontier, ops) = get_opfns_and_frontier(r);
    let state = state_builder(frontier)?;
    run_with_state_and_ops(&ops, state)
}

/// Run the circuit with a given starting state.
pub fn run_with_state<P: Precision, QS: QuantumState<P>>(
    r: &Register,
    state: QS,
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    let (frontier, ops) = get_opfns_and_frontier(r);

    let req_n = get_required_state_size::<P>(&frontier, &[]);

    if req_n != state.n() {
        let message = format!(
            "Circuit expected {:?} qubits but state contained {:?}",
            req_n,
            state.n()
        );
        CircuitError::make_err(message)
    } else {
        run_with_state_and_ops(&ops, state)
    }
}

/// `run` the pipeline using `LocalQuantumState`.
pub fn run_local<P: Precision>(
    r: &Register,
) -> Result<(LocalQuantumState<P>, MeasuredResults<P>), CircuitError> {
    run(r)
}

/// `run_with_init` the pipeline using `LocalQuantumState`
pub fn run_local_with_init<P: Precision>(
    r: &Register,
    states: &[RegisterInitialState<P>],
) -> Result<(LocalQuantumState<P>, MeasuredResults<P>), CircuitError> {
    run_with_init(r, states)
}

fn run_with_state_and_ops<P: Precision, QS: QuantumState<P>>(
    ops: &[&StateModifier],
    state: QS,
) -> Result<(QS, MeasuredResults<P>), CircuitError> {
    ops.iter()
        .cloned()
        .try_fold((state, MeasuredResults::new()), fold_modify_state)
}

/// Get the frontier of a circuit as well as references to all the StateModifiers needed in the
/// correct order.
pub fn get_opfns_and_frontier(r: &Register) -> (Vec<&Register>, Vec<&StateModifier>) {
    let mut heap = BinaryHeap::new();
    heap.push(r);
    let mut frontier_registers: Vec<&Register> = vec![];
    let mut fn_queue = VecDeque::new();
    while !heap.is_empty() {
        if let Some(r) = heap.pop() {
            match &r.parent {
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
                None => frontier_registers.push(r),
            }
            if let Some(deps) = &r.deps {
                deps.iter().for_each(|r| {
                    let r = r.as_ref();
                    if !in_heap(r, &heap) {
                        heap.push(r);
                    }
                })
            }
        }
    }
    (frontier_registers, fn_queue.into_iter().collect())
}

/// Deconstruct the circuit and own all the StateModifiers needed to run it.
pub fn get_owned_opfns(r: Register) -> Vec<StateModifier> {
    let mut heap = BinaryHeap::new();
    heap.push(r);
    let mut fn_queue = VecDeque::new();
    while !heap.is_empty() {
        if let Some(r) = heap.pop() {
            if let Some(parent) = r.parent {
                match parent {
                    Parent::Owned(parents, modifier) => {
                        if let Some(modifier) = modifier {
                            fn_queue.push_front(modifier);
                        }
                        heap.extend(parents);
                    }
                    Parent::Shared(r) => {
                        if let Ok(r) = Rc::try_unwrap(r) {
                            heap.push(r)
                        }
                    }
                }
            }
            if let Some(deps) = r.deps {
                deps.into_iter().for_each(|r| {
                    if let Ok(r) = Rc::try_unwrap(r) {
                        heap.push(r)
                    }
                })
            }
        }
    }
    fn_queue.into_iter().collect()
}

fn in_heap<T: Eq>(r: T, heap: &BinaryHeap<T>) -> bool {
    for hr in heap {
        if hr == &r {
            return true;
        }
    }
    false
}

/// Create a circuit for the circuit given by `r`. If `natural_order`, then the
/// qubit with index 0 represents the lowest bit in the index of the state (has the smallest
/// increment when flipped), otherwise it's the largest index (which is the internal state used by
/// the simulator).
pub fn make_circuit_matrix<P: Precision>(
    n: u64,
    r: &Register,
    natural_order: bool,
) -> Vec<Vec<Complex<P>>> {
    let indices: Vec<u64> = (0..n).collect();
    let lookup: Vec<Vec<Complex<P>>> = (0..1 << n)
        .map(|indx| {
            let indx = flip_bits(n as usize, indx);
            let (state, _) =
                run_local_with_init(r, &[(indices.clone(), InitialState::Index(indx))]).unwrap();
            (0..state.state.len())
                .map(|i| {
                    let indx = if natural_order {
                        flip_bits(n as usize, i as u64) as usize
                    } else {
                        i
                    };
                    state.state[indx]
                })
                .collect()
        })
        .collect();
    (0..1 << n)
        .map(|row| (0..1 << n).map(|col| lookup[col][row]).collect())
        .collect()
}
