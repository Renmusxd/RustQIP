extern crate num;
use crate::types::Precision;
use num::Complex;

/// A private version of QubitOp with variable precision, this is used so we can change the f64
/// default qubit op to a variable one at the beginning of execution and not at each operation.
pub enum PrecisionQubitOp<'a, P: Precision> {
    // Indices, Matrix data
    Matrix(Vec<u64>, Vec<Complex<P>>),
    // A indices, B indices
    Swap(Vec<u64>, Vec<u64>),
    // Control indices, Op indices, Op
    Control(Vec<u64>, Vec<u64>, Box<PrecisionQubitOp<'a, P>>),
    // Function which maps |x,y> to |x,f(x) xor y> where x,y are both m bits.
    Function(
        Vec<u64>,
        Vec<u64>,
        &'a (Fn(u64) -> (u64, f64) + Send + Sync),
    ),
}

/// Get the number of indices represented by `op`
pub fn precision_num_indices<P: Precision>(op: &PrecisionQubitOp<P>) -> usize {
    match &op {
        PrecisionQubitOp::Matrix(indices, _) => indices.len(),
        PrecisionQubitOp::Swap(a, b) => a.len() + b.len(),
        PrecisionQubitOp::Control(cs, os, _) => cs.len() + os.len(),
        PrecisionQubitOp::Function(inputs, outputs, _) => inputs.len() + outputs.len(),
    }
}

/// Get the `i`th qubit index for `op`
pub fn precision_get_index<P: Precision>(op: &PrecisionQubitOp<P>, i: usize) -> u64 {
    match &op {
        PrecisionQubitOp::Matrix(indices, _) => indices[i],
        PrecisionQubitOp::Swap(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        PrecisionQubitOp::Control(cs, os, _) => {
            if i < cs.len() {
                cs[i]
            } else {
                os[i - cs.len()]
            }
        }
        PrecisionQubitOp::Function(inputs, outputs, _) => {
            if i < inputs.len() {
                inputs[i]
            } else {
                outputs[i - inputs.len()]
            }
        }
    }
}