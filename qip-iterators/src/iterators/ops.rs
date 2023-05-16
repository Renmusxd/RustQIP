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
    /// A indices, B indices, each with n entries
    Swap(usize, Vec<usize>),
    /// (n) Control indices, Op indices, Op
    Control(usize, Vec<usize>, Box<MatrixOp<P>>),
}

impl<P> MatrixOp<P> {
    /// Get the number of indices represented by `op`
    pub fn num_indices(&self) -> usize {
        match self {
            MatrixOp::Matrix(indices, _) => indices.len(),
            MatrixOp::SparseMatrix(indices, _) => indices.len(),
            MatrixOp::Swap(a, b) => {
                debug_assert_eq!(b.len(), a*2);
                a*2
            },
            MatrixOp::Control(_, os, _) => {
                os.len()
            },
        }
    }

    /// Get the indices acted on by this op
    pub fn indices(&self) -> &[usize] {
        match self {
            MatrixOp::Matrix(i, _) => i,
            MatrixOp::SparseMatrix(i, _) => i,
            MatrixOp::Swap(_, i) => i,
            MatrixOp::Control(_, i, _) => i,
        }
    }

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
        let mut a = a.into();
        let n = a.len();
        let b = b.into();
        debug_assert_eq!(a.len(), b.len());
        a.extend(b);
        Self::Swap(n, a)
    }

    /// Make a new control matrix op
    pub fn new_control<IndxA, IndxB>(c: IndxA, r: IndxB, op: MatrixOp<P>) -> Self
    where
        IndxA: Into<Vec<usize>>,
        IndxB: Into<Vec<usize>>,
    {
        let mut c = c.into();
        let cn = c.len();
        let r = r.into();
        c.extend(r);
        Self::Control(cn, c, Box::new(op))
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
            MatrixOp::Control(n_control_indices, o_indices, op) => {
                let n_op_indices = o_indices.len() - n_control_indices;
                op.sum_for_control_iterator(row, *n_control_indices, n_op_indices, f)
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
            MatrixOp::Control(new_n_control_indices, o_indices, op) => {
                let n_control_indices = n_control_indices + new_n_control_indices;
                let n_op_indices = o_indices.len() - new_n_control_indices;
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
            MatrixOp::Swap(_n, indices) => {
                ("Swap".to_string(), indices.clone())
            }
            MatrixOp::Control(num_c_indices, indices, op) => {
                let name = format!("C({:?})", *op);
                (name, indices[..*num_c_indices].to_vec())
            }
        };
        let int_strings = indices
            .iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "{}[{}]", name, int_strings.join(", "))
    }
}
