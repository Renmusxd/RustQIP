use crate::iterators::{fold_for_op_cols, precision_get_index, precision_num_indices};
use crate::measurement_ops::MeasuredCondition;
use crate::pipeline::{get_initial_index_value_iterator, RegisterInitialState};
use crate::sparse_state::utils::{
    consolidate_vec, sparse_measure, sparse_measure_prob, sparse_measure_probs, sparse_soft_measure,
};
use crate::state_ops::{
    clone_as_precision_op, from_reals, full_to_sub, make_matrix_op, sub_to_full, transpose_op,
    UnitaryOp,
};
use crate::utils::flip_bits;
use crate::{Complex, Precision, QuantumState};
use num::{One, Zero};
use rayon::prelude::*;
use std::cmp::max;

/// A quantum state which doesn't track zero values.
#[derive(Debug)]
pub struct SparseQuantumState<P: Precision> {
    n: u64,
    state: Option<Vec<(u64, Complex<P>)>>,
    multithread: bool,
    input_region: Option<(u64, u64)>,
    output_region: Option<(u64, u64)>,
}

impl<P: Precision> SparseQuantumState<P> {
    /// Rotate to a new computational basis:
    /// `|0'> =  cos(angle)|0> + sin(angle)|1>`
    /// `|1'> = -sin(angle)|0> + cos(angle)|1>`
    pub fn rotate_basis(&mut self, indices: &[u64], angle: f64) {
        if angle != 0.0 {
            let (sangle, cangle) = angle.sin_cos();
            let basis_mat = from_reals(&[cangle, -sangle, sangle, cangle]);
            indices.iter().for_each(|indx| {
                let op = make_matrix_op(vec![*indx], basis_mat.clone()).unwrap();
                self.apply_op(&op);
            });
        }
    }

    /// Operate on state
    pub fn borrow_state<T, F: FnOnce(&Vec<(u64, Complex<P>)>) -> T>(&mut self, f: F) -> T {
        let s = self.state.take();
        let ret = if let Some(s) = &s {
            f(s)
        } else {
            panic!();
        };
        self.state = s;
        ret
    }

    fn make_initial_state(
        n: u64,
        states: &[RegisterInitialState<P>],
        input_offset: Option<(usize, usize)>,
    ) -> (u64, Vec<(u64, Complex<P>)>) {
        // If there's an index in the initial states above n, adjust n.
        let max_init_n = states
            .iter()
            .map(|(indices, _)| indices)
            .cloned()
            .flatten()
            .max()
            .map(|m| m + 1);
        let n = max_init_n.map(|m| max(n, m)).unwrap_or(n);

        let cvec =
            get_initial_index_value_iterator(n, states).fold(vec![], |mut acc, (indx, val)| {
                if val != Complex::zero() {
                    let should_push = match input_offset {
                        Some((low, high)) => (low <= indx) && ((indx) < high),
                        None => true,
                    };
                    if should_push {
                        acc.push((indx as u64, val));
                    }
                }
                acc
            });
        (n, cvec)
    }
}

impl<P: Precision> QuantumState<P> for SparseQuantumState<P> {
    fn new(n: u64) -> Self {
        Self {
            n,
            state: Some(vec![(0, Complex::one())]),
            multithread: true,
            input_region: None,
            output_region: None,
        }
    }

    fn new_from_initial_states(n: u64, states: &[RegisterInitialState<P>]) -> Self {
        let (n, cvec) = Self::make_initial_state(n, states, None);
        Self {
            n,
            state: Some(cvec),
            multithread: true,
            input_region: None,
            output_region: None,
        }
    }

    fn new_from_intitial_states_and_regions(
        n: u64,
        states: &[RegisterInitialState<P>],
        input_region: (usize, usize),
        output_region: (usize, usize),
    ) -> Self {
        let (n, cvec) = Self::make_initial_state(n, states, Some(input_region));

        // TODO clean up usage of u64. Should convert most things to usize.
        let input_region = (input_region.0 as u64, input_region.1 as u64);
        let output_region = (output_region.0 as u64, output_region.1 as u64);
        Self {
            n,
            state: Some(cvec),
            multithread: true,
            input_region: Some(input_region),
            output_region: Some(output_region),
        }
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn apply_op_with_name(&mut self, _name: Option<&str>, op: &UnitaryOp) {
        // More efficient for sparse to use transposed op.
        let op = transpose_op(op.clone());
        let op = clone_as_precision_op::<P>(&op);
        let mat_indices: Vec<u64> = (0..precision_num_indices(&op))
            .map(|i| precision_get_index(&op, i))
            .collect();
        let mat_mask = sub_to_full(self.n, &mat_indices, std::u64::MAX, 0);
        let nindices = precision_num_indices(&op) as u64;

        let state = self.state.as_ref().unwrap();
        let f = |(col, val): &(u64, Complex<P>)| -> Vec<(u64, Complex<P>)> {
            let matcol = full_to_sub(self.n, &mat_indices, *col);
            let col_template = (*col) & !mat_mask;
            fold_for_op_cols(nindices, matcol, &op, vec![], |mut acc, (row, row_val)| {
                let full_row = sub_to_full(self.n, &mat_indices, row, col_template);
                // Only output to valid locations.
                let should_push = match self.output_region {
                    Some((low, high)) => (low <= full_row) && (full_row < high),
                    None => true,
                };
                if should_push {
                    acc.push((full_row, val * row_val));
                }
                acc
            })
        };
        let flat = if self.multithread {
            state.par_iter().map(f).flatten().collect()
        } else {
            state.iter().map(f).flatten().collect()
        };
        self.state = Some(consolidate_vec(flat, self.multithread));
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P) {
        self.rotate_basis(indices, angle);
        let state = self.state.take().unwrap();
        let (measured_result, new_state) =
            sparse_measure(self.n, indices, state, measured, self.multithread);
        self.state = Some(new_state);
        self.rotate_basis(indices, -angle);
        measured_result
    }

    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, P) {
        self.rotate_basis(indices, angle);
        let state = self.state.as_ref().unwrap();
        let m = if let Some(m) = measured {
            m
        } else {
            sparse_soft_measure(self.n, indices, state, self.multithread)
        };
        let p = sparse_measure_prob(self.n, m, indices, state, self.multithread);
        self.rotate_basis(indices, -angle);
        (m, p)
    }

    fn state_magnitude(&self) -> P {
        let p: P = self
            .state
            .as_ref()
            .unwrap()
            .iter()
            .map(|(_, v)| v.norm_sqr())
            .sum();
        p.sqrt()
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        self.rotate_basis(indices, angle);
        let state = self.state.as_ref().unwrap();
        let probs = sparse_measure_probs(self.n, indices, state, self.multithread);
        self.rotate_basis(indices, -angle);
        probs
    }

    fn get_state(self, natural_order: bool) -> Vec<Complex<P>> {
        let mut state = vec![];
        let n = self.n as usize;
        state.resize(1 << n, Complex::zero());
        self.state.unwrap().into_iter().for_each(|(indx, val)| {
            let indx = if natural_order {
                flip_bits(n, indx)
            } else {
                indx
            };
            state[indx as usize] = val;
        });
        state
    }
}
