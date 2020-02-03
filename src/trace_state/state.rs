use crate::measurement_ops::MeasuredCondition;
use crate::pipeline::{InitialState, Representation};
use crate::sparse_state::SparseQuantumState;
use crate::state_ops::UnitaryOp;
use crate::utils::flip_bits;
use crate::{Complex, Precision, QuantumState};

/// A state which traces a single 1.0 entry and logs its position.
#[derive(Debug)]
pub struct TraceState<P: Precision> {
    /// Internal state.
    pub state: SparseQuantumState<P>,
    /// Trace over ops.
    pub trace: Vec<(Option<String>, u64)>,
}

impl<P: Precision> TraceState<P> {
    fn log_trace(&mut self, name: Option<&str>) {
        let n = self.n() as usize;
        let ret = self.state.borrow_state(|s| {
            if s.len() == 1 {
                let indx = s[0].0;
                let name = name.map(|s| s.to_string());
                (name, flip_bits(n, indx))
            } else {
                panic!();
            }
        });
        self.trace.push(ret);
    }
}

impl<P: Precision> QuantumState<P> for TraceState<P> {
    fn new(n: u64) -> Self {
        Self {
            state: SparseQuantumState::new(n),
            trace: Vec::new(),
        }
    }

    fn new_from_initial_states(n: u64, states: &[(Vec<u64>, InitialState<P>)]) -> Self {
        Self {
            state: SparseQuantumState::new_from_initial_states(n, states),
            trace: Vec::new(),
        }
    }

    fn n(&self) -> u64 {
        self.state.n()
    }

    fn apply_op_with_name(&mut self, name: Option<&str>, op: &UnitaryOp) {
        self.state.apply_op_with_name(name, op);
        self.log_trace(name);
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P) {
        let ret = self.state.measure(indices, measured, angle);
        self.log_trace(Some("Measure"));
        ret
    }

    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, P) {
        self.state.soft_measure(indices, measured, angle)
    }

    fn state_magnitude(&self) -> P {
        self.state.state_magnitude()
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        self.state.stochastic_measure(indices, angle)
    }

    fn into_state(self, order: Representation) -> Vec<Complex<P>> {
        self.state.into_state(order)
    }
}
