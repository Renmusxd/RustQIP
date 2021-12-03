#[cfg(feature = "parallel")]
pub(crate) use rayon::prelude::*;

use super::ops::UnitaryOp;
use super::qubit_iterators::*;
use crate::bridge;
use crate::types::Precision;
use crate::Complex;

/// Sums the outputs of `f` across all rows and values given by the iterator made from `op`
pub fn par_sum_from_iterator<T, F, P: Precision>(
    nindices: usize,
    row: usize,
    op: &UnitaryOp<P>,
    f: F,
) -> T
where
    T: std::iter::Sum + Send + Sync,
    F: Fn((usize, Complex<P>)) -> T + Send + Sync,
{
    match &op {
        UnitaryOp::Matrix(_, data) => {
            let iter = bridge!(MatrixOpIterator::new(row, nindices, data));
            iter.map(f).sum()
        }
        UnitaryOp::SparseMatrix(_, data) => {
            let iter = bridge!(SparseMatrixOpIterator::new(row, data));
            iter.map(f).sum()
        }
        UnitaryOp::Swap(_, _) => {
            let iter = bridge!(SwapOpIterator::new(row, nindices));
            iter.map(f).sum()
        }
        UnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = c_indices.len();
            let n_op_indices = o_indices.len();
            par_sum_from_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}

fn par_sum_from_control_iterator<T, F, P: Precision>(
    row: usize,
    op: &UnitaryOp<P>,
    n_control_indices: usize,
    n_op_indices: usize,
    f: F,
) -> T
where
    T: std::iter::Sum + Send + Sync,
    F: Fn((usize, Complex<P>)) -> T + Send + Sync,
{
    match &op {
        UnitaryOp::Matrix(_, data) => {
            let iter_builder = |row: usize| MatrixOpIterator::new(row, n_op_indices, data);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        UnitaryOp::SparseMatrix(_, data) => {
            let iter_builder = |row: usize| SparseMatrixOpIterator::new(row, data);
            let iter = bridge!(ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder
            ));
            iter.map(f).sum()
        }
        UnitaryOp::Swap(_, _) => {
            let iter_builder = |row: usize| SwapOpIterator::new(row, n_op_indices);
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
        UnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len();
            let n_op_indices = o_indices.len();
            par_sum_from_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}
