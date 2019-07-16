extern crate num;

use super::ops::PrecisionQubitOp;
use super::qubit_iterators::*;
use crate::types::Precision;
use num::Complex;
use crate::iterators::precision_num_indices;

/// Using the function `f` which maps from a column and `row` to a complex value for the op matrix,
/// sums for all nonzero entries for a given `op` more efficiently than trying each column between
/// 0 and 2^nindices.
/// This really needs to be cleaned up, but runs in a tight loop. This makes it hard since Box
/// is unfeasible and the iterator types aren't the same size.
pub fn sum_for_op_cols<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    nindices: u64,
    row: u64,
    ops: &[&PrecisionQubitOp<P>],
    f: F,
) -> Complex<P> {
    match &ops {
        [PrecisionQubitOp::Matrix(_, data)] => MatrixOpIterator::new(row, nindices, &data).map(f).sum(),
        [PrecisionQubitOp::Swap(_, _)] => SwapOpIterator::new(row, nindices).map(f).sum(),
        [PrecisionQubitOp::Control(c_indices, o_indices, op)] => {
            map_with_control_iterator(row, &op, c_indices.len() as u64, o_indices.len() as u64, f)
        },
        [PrecisionQubitOp::Function(inputs, outputs, op_f)] => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            FunctionOpIterator::new(row, input_n, output_n, op_f)
                .map(f)
                .sum()
        }
        [] => unimplemented!(),
        _ => {
            unimplemented!()
        }
    }
}

/// Builds a ControlledOpIterator for the given `op`, then maps using `f` and sums.
fn map_with_control_iterator<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    row: u64,
    op: &PrecisionQubitOp<P>,
    n_control_indices: u64,
    n_op_indices: u64,
    f: F,
) -> Complex<P> {
    match op {
        PrecisionQubitOp::Matrix(_, data) => {
            let iter_builder = |row: u64| MatrixOpIterator::new(row, n_op_indices, &data);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.map(f).sum()
        }
        PrecisionQubitOp::Swap(_, _) => {
            let iter_builder = |row: u64| SwapOpIterator::new(row, n_op_indices);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.map(f).sum()
        }
        PrecisionQubitOp::Function(inputs, outputs, op_f) => {
            let input_n = inputs.len() as u64;
            let output_n = outputs.len() as u64;
            let iter_builder = |row: u64| FunctionOpIterator::new(row, input_n, output_n, op_f);
            let it = ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder);
            it.map(f).sum()
        }
        // Control ops are automatically collapsed if made with helper, but implement this anyway
        // just to account for the possibility.
        PrecisionQubitOp::Control(c_indices, o_indices, op) => {
            let n_control_indices = n_control_indices + c_indices.len() as u64;
            let n_op_indices = o_indices.len() as u64;
            map_with_control_iterator(row, op, n_control_indices, n_op_indices, f)
        }
    }
}


fn sum_with_multi_op_iterator<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    nindices: u64,
    row: u64,
    ops: &[&PrecisionQubitOp<P>],
    mut iters: Vec<&mut Iterator<Item=(u64, Complex<P>)>>,
    f: F,
) -> Complex<P> {
    let some_tuple = ops.split_first();
    if let Some((op, ops)) = some_tuple {
        let nopindices = precision_num_indices(op) as u64;

        let mask = 1 << (nopindices + 1);
        let mask = mask - 1;

        let oprow = row & mask;
        let row = row >> nopindices;

        match op {
            PrecisionQubitOp::Matrix(indices, data) => {
                let mut it = MatrixOpIterator::new(row, nopindices, &data);
                iters.push(&mut it);
                let nindices = nindices - nopindices;
                sum_with_multi_op_iterator(nindices, row, ops, iters, f)
            },
            PrecisionQubitOp::Swap(a_indices, b_indices) => {
                let mut it = SwapOpIterator::new(row, nopindices);
                iters.push(&mut it);
                let nindices = nindices - nopindices;
                sum_with_multi_op_iterator(nindices, row, ops, iters, f)
            },
            PrecisionQubitOp::Control(c_indices, o_indices, op) => {
                unimplemented!()
//                map_with_control_iterator(row, &op, c_indices.len() as u64, o_indices.len() as u64, f)
            },
            PrecisionQubitOp::Function(inputs, outputs, op_f) => {
                let input_n = inputs.len() as u64;
                let output_n = outputs.len() as u64;
                let mut it = FunctionOpIterator::new(row, input_n, output_n, op_f);
                iters.push(&mut it);
                let nindices = nindices - nopindices;
                sum_with_multi_op_iterator(nindices, row, ops, iters, f)
            }
        }
    } else {
        unimplemented!()
    }
}

fn control_op_recursive_help<P: Precision, F: Fn((u64, Complex<P>)) -> Complex<P>>(
    nindices: u64,
    row: u64,
    ops: &[&PrecisionQubitOp<P>],
    mut iters: Vec<&Iterator<Item=(u64, Complex<P>)>>,
    f: F,
) -> Complex<P> {

}