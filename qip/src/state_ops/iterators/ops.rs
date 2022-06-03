use crate::types::Precision;
use num_complex::Complex;
use std::fmt;

/// Ops which can be applied to quantum states.
#[derive(Clone)]
pub enum UnitaryOp<P: Precision> {
    /// Indices, Matrix data
    Matrix(Vec<usize>, Vec<Complex<P>>),
    /// Indices, per row [(col, value)]
    SparseMatrix(Vec<usize>, Vec<Vec<(usize, Complex<P>)>>),
    /// A indices, B indices
    Swap(Vec<usize>, Vec<usize>),
    /// Control indices, Op indices, Op
    Control(Vec<usize>, Vec<usize>, Box<UnitaryOp<P>>),
}

impl<P: Precision> fmt::Debug for UnitaryOp<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (name, indices) = match self {
            UnitaryOp::Matrix(indices, _) => ("Matrix".to_string(), indices.clone()),
            UnitaryOp::SparseMatrix(indices, _) => ("SparseMatrix".to_string(), indices.clone()),
            UnitaryOp::Swap(a_indices, b_indices) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("Swap".to_string(), indices)
            }
            UnitaryOp::Control(indices, _, op) => {
                let name = format!("C({:?})", *op);
                (name, indices.clone())
            }
        };
        let int_strings = indices
            .iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "{}[{}]", name, int_strings.join(", "))
    }
}
