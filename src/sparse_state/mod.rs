/// State struct
pub mod state;
mod utils;

pub use state::SparseQuantumState;
use crate::{Precision, Register, CircuitError};
use crate::pipeline::{MeasuredResults, run, RegisterInitialState, run_with_init};


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
