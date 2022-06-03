use super::ops::UnitaryOp;
use super::qubit_iterators::*;
use crate::state_ops::iterators::MultiOpIterator;
use crate::state_ops::matrix_ops::num_indices;
use crate::types::Precision;
use crate::Complex;

/// Using the function `f` which maps from a column and `row` to a complex value for the op matrix,
/// sums for all nonzero entries for a given `op` more efficiently than trying each column between
/// 0 and 2^nindices.
/// This really needs to be cleaned up, but runs in a tight loop. This makes it hard since Box
/// is unfeasible and the iterator types aren't the same size.
pub fn sum_for_op_cols<T, P: Precision, F>(
    nindices: usize,
    row: usize,
    op: &UnitaryOp<P>,
    f: F,
) -> T
where
    T: std::iter::Sum,
    F: Fn((usize, Complex<P>)) -> T,
{
    act_on_iterator(nindices, row, op, move |iter| iter.map(f).sum())
}

/// Like `sum_for_op_cols` but takes multiple ops at once.
pub fn sum_for_ops_cols<P: Precision, F: Fn((usize, Complex<P>)) -> Complex<P>>(
    row: usize,
    ops: &[UnitaryOp<P>],
    f: F,
) -> Complex<P> {
    let ns = ops.iter().map(num_indices).collect::<Vec<_>>();
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
pub fn fold_for_op_cols<P: Precision, T, F: Fn(T, (usize, Complex<P>)) -> T>(
    nindices: usize,
    row: usize,
    op: &UnitaryOp<P>,
    init: T,
    f: F,
) -> T {
    act_on_iterator(nindices, row, op, move |iter| iter.fold(init, f))
}

/// Apply function f to the iterator for `op`, return result.
pub fn act_on_iterator<T, F, P: Precision>(
    nindices: usize,
    row: usize,
    op: &UnitaryOp<P>,
    f: F,
) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (usize, Complex<P>)>) -> T,
{
    match &op {
        UnitaryOp::Matrix(_, data) => f(&mut MatrixOpIterator::new(row, nindices, data)),
        UnitaryOp::SparseMatrix(_, data) => {
            f(&mut SparseMatrixOpIterator::new(row, data.as_slice()))
        }
        UnitaryOp::Swap(_, _) => f(&mut SwapOpIterator::new(row, nindices)),
        UnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = c_indices.len();
            let n_op_indices = o_indices.len();
            act_on_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}

fn act_on_control_iterator<T, F, P: Precision>(
    row: usize,
    op: &UnitaryOp<P>,
    n_control_indices: usize,
    n_op_indices: usize,
    f: F,
) -> T
where
    F: FnOnce(&mut dyn Iterator<Item = (usize, Complex<P>)>) -> T,
{
    match &op {
        UnitaryOp::Matrix(_, data) => {
            let iter_builder = |row: usize| MatrixOpIterator::new(row, n_op_indices, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        UnitaryOp::SparseMatrix(_, data) => {
            let iter_builder = |row: usize| SparseMatrixOpIterator::new(row, data);
            f(&mut ControlledOpIterator::new(
                row,
                n_control_indices,
                n_op_indices,
                iter_builder,
            ))
        }
        UnitaryOp::Swap(_, _) => {
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
        UnitaryOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len();
            let n_op_indices = o_indices.len();
            act_on_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}
