use super::ops::PrecisionUnitaryOp;
use super::qubit_iterators::*;
use crate::rayon_helper::*;
use crate::types::Precision;
use num::Complex;

/// Sums the outputs of `f` across all rows and values given by the iterator made from `op`
pub fn par_sum_from_iterator<T, F, P: Precision>(
    nindices: u64,
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    f: F,
) -> T
where
    T: std::iter::Sum + Send + Sync,
    F: Fn((u64, Complex<P>)) -> T + Send + Sync,
{
    match &op {
        PrecisionUnitaryOp::Matrix(_, data) => {
            let iter = bridge!(MatrixOpIterator::new(row, nindices, &data));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::SparseMatrix(_, data) => {
            let iter = bridge!(SparseMatrixOpIterator::new(row, &data));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::Swap(_, _) => {
            let iter = bridge!(SwapOpIterator::new(row, nindices));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            let iter = bridge!(FunctionOpIterator::new(row, input_n, output_n, op_f));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            par_sum_from_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}

fn par_sum_from_control_iterator<T, F, P: Precision>(
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    n_control_indices: u64,
    n_op_indices: u64,
    f: F,
) -> T
where
    T: std::iter::Sum + Send + Sync,
    F: Fn((u64, Complex<P>)) -> T + Send + Sync,
{
    match &op {
        PrecisionUnitaryOp::Matrix(_, data) => {
            let iter_builder = |row: u64| MatrixOpIterator::new(row, n_op_indices, &data);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::SparseMatrix(_, data) => {
            let iter_builder = |row: u64| SparseMatrixOpIterator::new(row, &data);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::Swap(_, _) => {
            let iter_builder = |row: u64| SwapOpIterator::new(row, n_op_indices);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        PrecisionUnitaryOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            let iter_builder = |row: u64| FunctionOpIterator::new(row, input_n, output_n, op_f);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        // Control ops are automatically collapsed if made with helper, but implement this anyway
        // just to account for the possibility.
        PrecisionUnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            par_sum_from_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}
