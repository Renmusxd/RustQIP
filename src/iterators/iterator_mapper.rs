extern crate num;

use super::ops::PrecisionQubitOp;
use super::qubit_iterators::*;
use crate::iterators::{precision_num_indices, MultiOpIterator};
use crate::types::Precision;
use num::Complex;

/// Using the function `f` which maps from a column and `row` to a complex value for the op matrix,
/// sums for all nonzero entries for a given `op` more efficiently than trying each column between
/// 0 and 2^nindices.
/// This really needs to be cleaned up, but runs in a tight loop. This makes it hard since Box
/// is unfeasible and the iterator types aren't the same size.
pub fn sum_for_op_cols<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    nindices: u64,
    row: u64,
    op: &PrecisionQubitOp<P>,
    f: F,
) -> Complex<P> {
    let init = Complex {
        re: P::zero(),
        im: P::zero(),
    };
    fold_for_op_cols(nindices, row, op, init, |acc, entry| acc + f(entry))
}

pub fn sum_for_ops_cols<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    row: u64,
    ops: &[PrecisionQubitOp<P>],
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
                let acc_row = acc_row >> *op_nindices as u64;
                (acc, acc_row)
            });
    let v_slices: Vec<_> = v.iter().map(|v| v.as_slice()).collect();

    let it = MultiOpIterator::new(&ns, &v_slices);
    it.map(f).sum()
}

pub fn fold_for_op_cols<P: Precision, T, F: Fn(T, (u64, Complex<P>)) -> T>(
    nindices: u64,
    row: u64,
    op: &PrecisionQubitOp<P>,
    init: T,
    f: F,
) -> T {
    match &op {
        PrecisionQubitOp::Matrix(_, data) => {
            MatrixOpIterator::new(row, nindices, &data).fold(init, f)
        }
        PrecisionQubitOp::SparseMatrix(_, data) => {
            SparseMatrixOpIterator::new(row, &data).fold(init, f)
        }
        PrecisionQubitOp::Swap(_, _) => SwapOpIterator::new(row, nindices).fold(init, f),
        PrecisionQubitOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            fold_with_control_iterator(row, &op, n_control_indices, n_op_indices, init, f)
        }
        PrecisionQubitOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            FunctionOpIterator::new(row, input_n, output_n, op_f).fold(init, f)
        }
    }
}

/// Builds a ControlledOpIterator for the given `op`, then maps using `f` and sums.
fn fold_with_control_iterator<P: Precision, T, F: Fn(T, (u64, Complex<P>)) -> T>(
    row: u64,
    op: &PrecisionQubitOp<P>,
    n_control_indices: u64,
    n_op_indices: u64,
    init: T,
    f: F,
) -> T {
    match op {
        PrecisionQubitOp::Matrix(_, data) => {
            let iter_builder = |row: u64| MatrixOpIterator::new(row, n_op_indices, &data);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.fold(init, f)
        }
        PrecisionQubitOp::SparseMatrix(_, data) => {
            let iter_builder = |row: u64| SparseMatrixOpIterator::new(row, &data);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.fold(init, f)
        }
        PrecisionQubitOp::Swap(_, _) => {
            let iter_builder = |row: u64| SwapOpIterator::new(row, n_op_indices);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.fold(init, f)
        }
        PrecisionQubitOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            let iter_builder = |row: u64| FunctionOpIterator::new(row, input_n, output_n, op_f);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.fold(init, f)
        }
        // Control ops are automatically collapsed if made with helper, but implement this anyway
        // just to account for the possibility.
        PrecisionQubitOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            fold_with_control_iterator(row, op, n_control_indices, n_op_indices, init, f)
        }
    }
}
