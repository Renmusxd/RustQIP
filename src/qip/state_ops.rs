extern crate num;

use num::complex::Complex;

pub type OperatorFn = fn(QuantumState) -> QuantumState;

pub enum QubitOp {
    MatrixOp(usize, Vec<Complex<f64>>),
    SwapOp(Vec<u64>, Vec<u64>),
    ControlOp(u64, Box<QubitOp>),
}

pub struct QuantumState {
    // A bundle with the quantum state data.
}


pub fn to_complex(data: &[f64]) -> Vec<Complex<f64>> {
    data.into_iter().map(|x| Complex::<f64> {
        re: x.clone(),
        im: 0.0
    }).collect()
}

pub fn make_op_fn(op: QubitOp) -> OperatorFn {
    |x| x
}