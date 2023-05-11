use crate::iterators::{
    ControlledOpIterator, MatrixOpIterator, SparseMatrixOpIterator, SwapOpIterator,
};
use num_traits::{One, Zero};
use std::fmt;
use std::iter::Sum;
use std::ops::Mul;

/// Ops which can be applied to quantum states.
#[derive(Clone)]
pub enum MatrixOp<P> {
    /// Indices, Matrix data
    Matrix(Vec<usize>, Vec<P>),
    /// Indices, per row [(col, value)]
    SparseMatrix(Vec<usize>, Vec<Vec<(usize, P)>>),
    /// A indices, B indices
    Swap(Vec<usize>, Vec<usize>),
    /// Control indices, Op indices, Op
    Control(Vec<usize>, Vec<usize>, Box<MatrixOp<P>>),
}

impl<P> MatrixOp<P> {
    /// Make a new dense matrix op
    pub fn new_matrix<Indx, Dat>(indices: Indx, data: Dat) -> Self
    where
        Indx: Into<Vec<usize>>,
        Dat: Into<Vec<P>>,
    {
        Self::Matrix(indices.into(), data.into())
    }

    /// Make a new sparse matrix op
    pub fn new_sparse<Indx, Dat>(indices: Indx, data: Dat) -> Self
    where
        Indx: Into<Vec<usize>>,
        Dat: Into<Vec<Vec<(usize, P)>>>,
    {
        Self::SparseMatrix(indices.into(), data.into())
    }

    /// Make a new swap matrix op
    pub fn new_swap<IndxA, IndxB>(a: IndxA, b: IndxB) -> Self
    where
        IndxA: Into<Vec<usize>>,
        IndxB: Into<Vec<usize>>,
    {
        Self::Swap(a.into(), b.into())
    }

    /// Make a new control matrix op
    pub fn new_control<IndxA, IndxB>(c: IndxA, r: IndxB, op: MatrixOp<P>) -> Self
    where
        IndxA: Into<Vec<usize>>,
        IndxB: Into<Vec<usize>>,
    {
        Self::Control(c.into(), r.into(), Box::new(op))
    }
}

impl<P> MatrixOp<P>
where
    P: Clone + Zero + One + Mul,
{
    /// The function `f` maps a column to a complex value (given the `row`) for the op matrix.
    /// Sums for all nonzero entries for a given `op`
    pub fn sum_for_op_cols<T, F>(&self, nindices: usize, row: usize, f: F) -> T
    where
        T: Sum,
        F: Fn((usize, P)) -> T,
    {
        match &self {
            MatrixOp::Matrix(_, data) => MatrixOpIterator::new(row, nindices, data).map(f).sum(),
            MatrixOp::SparseMatrix(_, data) => SparseMatrixOpIterator::new(row, data.as_slice())
                .map(f)
                .sum(),
            MatrixOp::Swap(_, _) => SwapOpIterator::new(row, nindices).map(f).sum(),
            MatrixOp::Control(c_indices, o_indices, op) => {
                let n_control_indices = c_indices.len();
                let n_op_indices = o_indices.len();
                op.sum_for_control_iterator(row, n_control_indices, n_op_indices, f)
            }
        }
    }

    fn sum_for_control_iterator<T, F>(
        &self,
        row: usize,
        n_control_indices: usize,
        n_op_indices: usize,
        f: F,
    ) -> T
    where
        T: Sum,
        F: Fn((usize, P)) -> T,
    {
        match &self {
            MatrixOp::Matrix(_, data) => {
                let iter_builder = |row: usize| MatrixOpIterator::new(row, n_op_indices, data);
                ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder)
                    .map(f)
                    .sum()
            }
            MatrixOp::SparseMatrix(_, data) => {
                let iter_builder = |row: usize| SparseMatrixOpIterator::new(row, data);
                ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder)
                    .map(f)
                    .sum()
            }
            MatrixOp::Swap(_, _) => {
                let iter_builder = |row: usize| SwapOpIterator::new(row, n_op_indices);
                ControlledOpIterator::new(row, n_control_indices, n_op_indices, iter_builder)
                    .map(f)
                    .sum()
            }
            // Control ops are automatically collapsed if made with helper, but implement this anyway
            // just to account for the possibility.
            MatrixOp::Control(c_indices, o_indices, op) => {
                let n_control_indices = n_control_indices + c_indices.len();
                let n_op_indices = o_indices.len();
                op.sum_for_control_iterator(row, n_control_indices, n_op_indices, f)
            }
        }
    }
}

impl<P> fmt::Debug for MatrixOp<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (name, indices) = match self {
            MatrixOp::Matrix(indices, _) => ("Matrix".to_string(), indices.clone()),
            MatrixOp::SparseMatrix(indices, _) => ("SparseMatrix".to_string(), indices.clone()),
            MatrixOp::Swap(a_indices, b_indices) => {
                let indices: Vec<_> = a_indices
                    .iter()
                    .cloned()
                    .chain(b_indices.iter().cloned())
                    .collect();
                ("Swap".to_string(), indices)
            }
            MatrixOp::Control(indices, _, op) => {
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

/// Get the number of indices represented by `op`
pub fn num_indices<P>(op: &MatrixOp<P>) -> usize {
    match &op {
        MatrixOp::Matrix(indices, _) => indices.len(),
        MatrixOp::SparseMatrix(indices, _) => indices.len(),
        MatrixOp::Swap(a, b) => a.len() + b.len(),
        MatrixOp::Control(cs, os, _) => cs.len() + os.len(),
    }
}
