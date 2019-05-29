extern crate num;
use super::qubits::Qubit;
use num::complex::Complex;

pub type OperatorFn = fn(Box<QuantumState>) -> Box<QuantumState>;
pub type MeasuredResultReference = u32;

pub enum QubitOp {
    MatrixOp(usize, Vec<Complex<f64>>),
    SwapOp(Vec<u64>, Vec<u64>),
    ControlOp(Vec<u64>, Box<QubitOp>),
}

pub fn make_op_fn(op: QubitOp) -> OperatorFn {
    |x| x
}

pub trait QuantumState {

}

struct LocalQuantumState {
    // A bundle with the quantum state data.
    state: Vec<Complex<f64>>,
    arena: Vec<Complex<f64>>,
}

impl QuantumState for LocalQuantumState {

}

pub fn run(q: &Qubit) {
    unimplemented!()
}