/// State struct
pub mod state;
mod utils;

use crate::pipeline::{run, run_with_init, MeasuredResults, RegisterInitialState};
use crate::{CircuitError, Precision, Register};
pub use state::SparseQuantumState;

/// `run` the pipeline using `SparseQuantumState`.
pub fn run_sparse_local<P: Precision>(
    r: &Register,
) -> Result<(SparseQuantumState<P>, MeasuredResults<P>), CircuitError> {
    run(r)
}

/// `run_with_init` the pipeline using `SparseQuantumState`
pub fn run_sparse_local_with_init<P: Precision>(
    r: &Register,
    states: &[RegisterInitialState<P>],
) -> Result<(SparseQuantumState<P>, MeasuredResults<P>), CircuitError> {
    run_with_init(r, states)
}
