use crate::iterators::*;
use num_traits::{One, Zero};
use std::iter::Sum;
use std::ops::Mul;

/// Like `sum_for_op_cols` but takes multiple ops at once.
/// TODO: Clean up, remove fold_for_op_cols
pub fn sum_for_ops_cols<P, F: Fn((usize, P)) -> P>(row: usize, ops: &[MatrixOp<P>], f: F) -> P
where
    P: Sum + Clone + Mul + One + Zero,
{
    let ns = ops.iter().map(MatrixOp::num_indices).collect::<Vec<_>>();
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
pub fn fold_for_op_cols<P, T, F: Fn(T, (usize, P)) -> T>(
    nindices: usize,
    row: usize,
    op: &MatrixOp<P>,
    init: T,
    f: F,
) -> T
where
    P: Clone + Zero + One,
{
    act_on_iterator(nindices, row, op, move |iter| iter.fold(init, f))
}

/// Apply function f to the iterator for `op`, return result.
pub fn act_on_iterator<T, F, P>(nindices: usize, row: usize, op: &MatrixOp<P>, f: F) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (usize, P)>) -> T,
    P: Clone + Zero + One,
{
    match &op {
        MatrixOp::Matrix(_, data) => f(&mut MatrixOpIterator::new(row, nindices, data)),
        MatrixOp::SparseMatrix(_, data) => {
            f(&mut SparseMatrixOpIterator::new(row, data.as_slice()))
        }
        MatrixOp::Swap(_, _) => f(&mut SwapOpIterator::new(row, nindices)),
        MatrixOp::Control(n_control_indices, o_indices, op) => {
            let n_op_indices = o_indices.len() - n_control_indices;
            act_on_control_iterator(row, op, *n_control_indices, n_op_indices, f)
        }
    }
}

fn act_on_control_iterator<T, F, P>(
    row: usize,
    op: &MatrixOp<P>,
    n_control_indices: usize,
    n_op_indices: usize,
    f: F,
) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (usize, P)>) -> T,
    P: Clone + One + Zero + Mul,
{
    match &op {
        MatrixOp::Matrix(_, data) => {
            let iter_builder = |row: usize| MatrixOpIterator::new(row, n_op_indices, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        MatrixOp::SparseMatrix(_, data) => {
            let iter_builder = |row: usize| SparseMatrixOpIterator::new(row, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        MatrixOp::Swap(_, _) => {
            let iter_builder = |row: usize| SwapOpIterator::new(row, n_op_indices);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        // Control ops are automatically collapsed if made with helper, but implement this anyway
        // just to account for the possibility.
        MatrixOp::Control(new_n_control_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + new_n_control_indices;
            let n_op_indices = o_indices.len() - new_n_control_indices;
            act_on_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}
