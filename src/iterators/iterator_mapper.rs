use super::ops::PrecisionUnitaryOp;
use super::qubit_iterators::*;
use crate::iterators::{precision_num_indices, MultiOpIterator};
use crate::types::Precision;
use num::Complex;

/// Using the function `f` which maps from a column and `row` to a complex value for the op matrix,
/// sums for all nonzero entries for a given `op` more efficiently than trying each column between
/// 0 and 2^nindices.
/// This really needs to be cleaned up, but runs in a tight loop. This makes it hard since Box
/// is unfeasible and the iterator types aren't the same size.
pub fn sum_for_op_cols<T, P: Precision, F>(
    nindices: u64,
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    f: F,
) -> T
where
    T: std::iter::Sum,
    F: Fn((u64, Complex<P>)) -> T,
{
    act_on_iterator(nindices, row, op, move |iter| iter.map(f).sum())
}

/// Like `sum_for_op_cols` but takes multiple ops at once.
pub fn sum_for_ops_cols<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    row: u64,
    ops: &[PrecisionUnitaryOp<P>],
    f: F,
) -> Complex<P> {
    let ns: Vec<_> = ops
        .iter()
        .map(|op| precision_num_indices(op) as u64)
        .collect();
    let (v, _) =
        ops.iter()
            .zip(ns.iter())
            .fold((vec![], row), |(mut acc, acc_row), (op, op_nindices)| {
                let mask = (1 << *op_nindices) - 1;
                let op_row = acc_row & mask;
                let v = fold_for_op_cols(*op_nindices, op_row, op, vec![], |mut acc, entry| {
                    acc.push(entry);
                    acc
                });
                acc.push(v);
                let acc_row = acc_row >> *op_nindices;
                (acc, acc_row)
            });
    let v_slices: Vec<_> = v.iter().map(|v| v.as_slice()).collect();

    let it = MultiOpIterator::new(&ns, &v_slices);
    it.map(f).sum()
}

/// Fold across rows hitting nonzero columns.
pub fn fold_for_op_cols<P: Precision, T, F: Fn(T, (u64, Complex<P>)) -> T>(
    nindices: u64,
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    init: T,
    f: F,
) -> T {
    act_on_iterator(nindices, row, op, move |iter| iter.fold(init, f))
}

/// Apply function f to the iterator for `op`, return result.
pub fn act_on_iterator<T, F, P: Precision>(
    nindices: u64,
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    f: F,
) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (u64, Complex<P>)>) -> T,
{
    match &op {
        PrecisionUnitaryOp::Matrix(_, data) => f(&mut MatrixOpIterator::new(row, nindices, data)),
        PrecisionUnitaryOp::SparseMatrix(_, data) => {
            f(&mut SparseMatrixOpIterator::new(row, data))
        }
        PrecisionUnitaryOp::Swap(_, _) => f(&mut SwapOpIterator::new(row, nindices)),
        PrecisionUnitaryOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            f(&mut FunctionOpIterator::new(row, input_n, output_n, op_f))
        }
        PrecisionUnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            act_on_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}

fn act_on_control_iterator<T, F, P: Precision>(
    row: u64,
    op: &PrecisionUnitaryOp<P>,
    n_control_indices: u64,
    n_op_indices: u64,
    f: F,
) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (u64, Complex<P>)>) -> T,
{
    match &op {
        PrecisionUnitaryOp::Matrix(_, data) => {
            let iter_builder = |row: u64| MatrixOpIterator::new(row, n_op_indices, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        PrecisionUnitaryOp::SparseMatrix(_, data) => {
            let iter_builder = |row: u64| SparseMatrixOpIterator::new(row, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        PrecisionUnitaryOp::Swap(_, _) => {
            let iter_builder = |row: u64| SwapOpIterator::new(row, n_op_indices);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        PrecisionUnitaryOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            let iter_builder = |row: u64| FunctionOpIterator::new(row, input_n, output_n, op_f);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        // Control ops are automatically collapsed if made with helper, but implement this anyway
        // just to account for the possibility.
        PrecisionUnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            act_on_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}
