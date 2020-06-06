use crate::types::Precision;
use num::Complex;
use std::fmt;

/// A private version of UnitaryOp with variable precision, this is used so we can change the f64
/// default UnitaryOp to a variable one at the beginning of execution and not at each operation.
pub enum PrecisionUnitaryOp<'a, P: Precision> {
    /// Indices, Matrix data
    Matrix(Vec<u64>, Vec<Complex<P>>),
    /// Indices, per row [(col, value)]
    SparseMatrix(Vec<u64>, Vec<Vec<(u64, Complex<P>)>>),
    /// A indices, B indices
    Swap(Vec<u64>, Vec<u64>),
    /// Control indices, Op indices, Op
    Control(Vec<u64>, Vec<u64>, Box<PrecisionUnitaryOp<'a, P>>),
    /// Function which maps |x,y> to |x,f(x) xor y> where x,y are both m bits.
    Function(
        Vec<u64>,
        Vec<u64>,
        &'a (dyn Fn(u64) -> (u64, f64) + Send + Sync),
    ),
}

impl<'a, P: Precision> fmt::Debug for PrecisionUnitaryOp<'a, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (name, indices) = match self {
            PrecisionUnitaryOp::Matrix(indices, _) => ("Matrix".to_string(), indices.clone()),
            PrecisionUnitaryOp::SparseMatrix(indices, _) => {
                ("SparseMatrix".to_string(), indices.clone())
            }
            PrecisionUnitaryOp::Swap(a_indices, b_indices) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("Swap".to_string(), indices)
            }
            PrecisionUnitaryOp::Control(indices, _, op) => {
                let name = format!("C({:?})", *op);
                (name, indices.clone())
            }
            PrecisionUnitaryOp::Function(a_indices, b_indices, _) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("F".to_string(), indices)
            }
        };
        let int_strings = indices
            .iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "{}[{}]", name, int_strings.join(", "))
    }
}

/// Get the number of indices represented by `op`
pub fn precision_num_indices<P: Precision>(op: &PrecisionUnitaryOp<P>) -> usize {
    match &op {
        PrecisionUnitaryOp::Matrix(indices, _) => indices.len(),
        PrecisionUnitaryOp::SparseMatrix(indices, _) => indices.len(),
        PrecisionUnitaryOp::Swap(a, b) => a.len() + b.len(),
        PrecisionUnitaryOp::Control(cs, os, _) => cs.len() + os.len(),
        PrecisionUnitaryOp::Function(inputs, outputs, _) => inputs.len() + outputs.len(),
    }
}

/// Get the `i`th qubit index for `op`
pub fn precision_get_index<P: Precision>(op: &PrecisionUnitaryOp<P>, i: usize) -> u64 {
    match &op {
        PrecisionUnitaryOp::Matrix(indices, _) => indices[i],
        PrecisionUnitaryOp::SparseMatrix(indices, _) => indices[i],
        PrecisionUnitaryOp::Swap(a, b) => {
            if i < a.len() {
                a[i]
            } else {
                b[i - a.len()]
            }
        }
        PrecisionUnitaryOp::Control(cs, os, _) => {
            if i < cs.len() {
                cs[i]
            } else {
                os[i - cs.len()]
            }
        }
        PrecisionUnitaryOp::Function(inputs, outputs, _) => {
            if i < inputs.len() {
                inputs[i]
            } else {
                outputs[i - inputs.len()]
            }
        }
    }
}
