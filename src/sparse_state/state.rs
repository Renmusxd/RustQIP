use crate::iterators::{fold_for_op_cols, precision_get_index, precision_num_indices};
use crate::measurement_ops::MeasuredCondition;
use crate::pipeline::{create_state_entry, InitialState, Representation};
use crate::rayon_helper::*;
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
use std::cmp::max;

/// A quantum state which doesn't track zero values.
#[derive(Debug)]
pub struct SparseQuantumState<P: Precision> {
    n: u64,
    state: Option<Vec<(u64, Complex<P>)>>,
    multithread: bool,
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
}

impl<P: Precision> QuantumState<P> for SparseQuantumState<P> {
    fn new(n: u64) -> Self {
        Self {
            n,
            state: Some(vec![(0, Complex::one())]),
            multithread: true,
        }
    }

    fn new_from_initial_states(n: u64, states: &[(Vec<u64>, InitialState<P>)]) -> Self {
        let max_init_n = states
            .iter()
            .map(|(indices, _)| indices)
            .cloned()
            .flatten()
            .max()
            .map(|m| m + 1);
        let n = max_init_n.map_or(n, |m| max(n, m));

        // Assume that all unrepresented indices are in the |0> state.
        let n_fullindices: u64 = states
            .iter()
            .map(|(indices, state)| match state {
                InitialState::FullState(_) => indices.len() as u64,
                _ => 0,
            })
            .sum();

        // Make the index template/base
        let template: u64 = states.iter().fold(0, |acc, (indices, state)| -> u64 {
            match state {
                InitialState::Index(val_indx) => {
                    let val_indx = flip_bits(indices.len(), *val_indx);
                    sub_to_full(n, indices, val_indx, acc)
                }
                _ => acc,
            }
        });

        // Go through each combination of full index locations
        let cvec = (0..1 << n_fullindices).fold(vec![], |mut acc, i| {
            let (delta_index, val) = create_state_entry(n, i, states);
            if val != Complex::zero() {
                acc.push((delta_index + template, val));
            }
            acc
        });

        Self {
            n,
            state: Some(cvec),
            multithread: true,
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
                acc.push((full_row, val * row_val));
                acc
            })
        };

        let flat = iter!(state).map(f).flatten().collect();
        self.state = Some(consolidate_vec(flat));
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P) {
        self.rotate_basis(indices, angle);
        let state = self.state.take().unwrap();
        let (measured_result, new_state) = sparse_measure(self.n, indices, state, measured);
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
            sparse_soft_measure(self.n, indices, state)
        };
        let p = sparse_measure_prob(self.n, m, indices, state);
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
        let probs = sparse_measure_probs(self.n, indices, state);
        self.rotate_basis(indices, -angle);
        probs
    }

    fn into_state(self, order: Representation) -> Vec<Complex<P>> {
        let mut state = vec![];
        let n = self.n as usize;
        state.resize(1 << n, Complex::zero());
        self.state.unwrap().into_iter().for_each(|(indx, val)| {
            let indx = match order {
                Representation::LittleEndian => flip_bits(n, indx),
                Representation::BigEndian => indx,
            };
            state[indx as usize] = val;
        });
        state
    }
}
