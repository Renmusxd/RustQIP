use crate::feynman_state::memoization::FeynmanMemory;
use crate::iterators::{
    par_sum_from_iterator, precision_get_index, precision_num_indices, sum_for_op_cols,
    PrecisionUnitaryOp,
};
use crate::measurement_ops::{measure_prob_fn, MeasuredCondition};
use crate::pipeline::{InitialState, Representation};
use crate::rayon_helper::*;
use crate::state_ops::{clone_as_precision_op, full_to_sub, sub_to_full, UnitaryOp};
use crate::utils::{extract_bits, flip_bits};
use crate::{Precision, QuantumState};
use num::{Complex, Zero};

const DEFAULT_DEPTH: u64 = 5;

/// The FeynmanState backend calculates state amplitudes using the feynman path integral
/// which means it is exponential with circuit depth, but has memory overhead linear with circuit
/// depth and independent of number of qubits. This means this state is good for large but shallow
/// circuits. If m is the number of qubits which have ops applied, allowing each qubit to be counted
/// multiple time, then the runtime is O(2^m).
#[derive(Debug)]
pub struct FeynmanState<P: Precision> {
    ops: Vec<FeynmanOp<P>>,
    parallel_depth: u64,
    substate: FeynmanThreadSafeState<P>,
    memory_size: usize,
}

#[derive(Debug)]
struct FeynmanThreadSafeState<P: Precision> {
    n: u64,
    mag: P,
    input_offset: u64,
    output_offset: u64,
    initial_state: Vec<(Vec<u64>, InitialState<P>)>,
    noninit_mask: u64,
}

#[derive(Debug)]
pub(crate) enum FeynmanOp<P: Precision> {
    OP(UnitaryOp),
    MEASUREMENT(u64, Vec<u64>, P),
}

#[derive(Debug)]
pub(crate) enum FeynmanPrecisionOp<'a, P: Precision> {
    OP(PrecisionUnitaryOp<'a, P>, Vec<u64>),
    MEASUREMENT(u64, Vec<u64>, P),
}

impl<P: Precision> FeynmanState<P> {
    fn new_from_initial_states_and_depth(
        n: u64,
        states: &[(Vec<u64>, InitialState<P>)],
        parallel_depth: u64,
    ) -> Self {
        let mag = states.iter().map(|(_, s)| s.get_magnitude()).product();

        let mask = states.iter().fold(u64::MAX, |mask, (indices, _)| {
            let submask = !sub_to_full(n, indices, u64::MAX, 0);
            mask & submask
        });

        let substate = FeynmanThreadSafeState {
            n,
            mag,
            input_offset: 0,
            output_offset: 0,
            initial_state: states.to_vec(),
            noninit_mask: mask,
        };
        Self {
            ops: vec![],
            parallel_depth,
            substate,
            memory_size: 0,
        }
    }

    fn make_precision_ops(&self) -> Vec<FeynmanPrecisionOp<P>> {
        self.ops
            .iter()
            .map(|op| match op {
                FeynmanOp::OP(op) => {
                    let pop = clone_as_precision_op(op);
                    let mat_indices: Vec<u64> = (0..precision_num_indices(&pop))
                        .map(|i| precision_get_index(&pop, i))
                        .collect();
                    FeynmanPrecisionOp::OP(pop, mat_indices)
                }
                FeynmanOp::MEASUREMENT(m, indices, p) => {
                    FeynmanPrecisionOp::MEASUREMENT(*m, indices.clone(), *p)
                }
            })
            .collect()
    }

    fn make_memory_container(
        &self,
        pops: &[FeynmanPrecisionOp<P>],
    ) -> Option<(usize, FeynmanMemory<P>)> {
        if self.memory_size > 0 && self.ops.len() > 1 {
            let memory_depth = pops.len() / 2;
            let sub_ops = &pops[..memory_depth];
            let mut mem = FeynmanMemory::<P>::new(self.memory_size, sub_ops);
            // TODO consider doing this recursively to cut down total runtime.
            mem.iter_mut().for_each(|(i, c)| {
                *c = self.substate.rec_calculate_amplitude(
                    i as u64,
                    sub_ops,
                    self.parallel_depth,
                    &None,
                );
            });
            Some((memory_depth, mem))
        } else {
            None
        }
    }

    /// Calculate the amplitude of a given state.
    pub fn calculate_amplitude(&self, m: u64) -> Complex<P> {
        let pops = self.make_precision_ops();
        let memory = self.make_memory_container(&pops);
        self.substate
            .rec_calculate_amplitude(m, &pops, self.parallel_depth, &memory)
    }
}

impl<P: Precision> FeynmanThreadSafeState<P> {
    fn rec_calculate_amplitude(
        &self,
        m: u64,
        ops: &[FeynmanPrecisionOp<P>],
        parallel_depth: u64,
        memory: &Option<(usize, FeynmanMemory<P>)>,
    ) -> Complex<P> {
        if let Some((memory_depth, memory)) = memory {
            if ops.len() == *memory_depth {
                if let Some(result) = memory.get(m as usize) {
                    return *result;
                }
            }
        }

        match ops {
            [] => {
                // Check that the uninitialized bits are 0
                if m & self.noninit_mask == 0 {
                    // Get values from initial state
                    self.initial_state
                        .iter()
                        .map(|(indices, init)| {
                            let subindex = full_to_sub(self.n, indices, m);
                            init.get_amplitude(subindex)
                        })
                        .product()
                } else {
                    Complex::zero()
                }
            }
            slice => {
                let head = &slice[0..slice.len() - 1];
                match &slice[slice.len() - 1] {
                    FeynmanPrecisionOp::OP(op, mat_indices) => {
                        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
                        // for the row calculated above.
                        let next_depth = if parallel_depth > 0 {
                            parallel_depth - 1
                        } else {
                            0
                        };
                        let f = |(i, val): (u64, Complex<P>)| -> Complex<P> {
                            let colbits = sub_to_full(self.n, mat_indices, i, m);
                            val * self.rec_calculate_amplitude(colbits, head, next_depth, memory)
                        };

                        let nindices = mat_indices.len() as u64;
                        let matrow = full_to_sub(self.n, mat_indices, m);

                        // We only want to allow parallelism for the top level.
                        if parallel_depth > 0 {
                            par_sum_from_iterator(nindices, matrow, op, f)
                        } else {
                            sum_for_op_cols(nindices, matrow, op, f)
                        }
                    }
                    FeynmanPrecisionOp::MEASUREMENT(measured, indices, p) => {
                        if *measured == full_to_sub(self.n, indices, m) {
                            self.rec_calculate_amplitude(m, head, parallel_depth, memory) / p.sqrt()
                        } else {
                            Complex::zero()
                        }
                    }
                }
            }
        }
    }
}

impl<P: Precision> QuantumState<P> for FeynmanState<P> {
    fn new(n: u64) -> Self {
        Self::new_from_initial_states(n, &[])
    }

    fn new_from_initial_states(n: u64, states: &[(Vec<u64>, InitialState<P>)]) -> Self {
        Self::new_from_initial_states_and_depth(n, states, DEFAULT_DEPTH)
    }

    fn n(&self) -> u64 {
        self.substate.n
    }

    fn apply_op_with_name(&mut self, _name: Option<&str>, op: &UnitaryOp) {
        self.ops.push(FeynmanOp::OP(op.clone()))
    }

    fn measure(
        &mut self,
        indices: &[u64],
        measured: Option<MeasuredCondition<P>>,
        angle: f64,
    ) -> (u64, P) {
        let (m, p) = match measured {
            Some(MeasuredCondition { measured, prob }) => {
                if let Some(prob) = prob {
                    (measured, prob)
                } else {
                    self.soft_measure(indices, Some(measured), angle)
                }
            }
            None => self.soft_measure(indices, None, angle),
        };
        self.ops
            .push(FeynmanOp::MEASUREMENT(m, indices.to_vec(), p));
        (m, p)
    }

    fn soft_measure(&mut self, indices: &[u64], measured: Option<u64>, angle: f64) -> (u64, P) {
        self.rotate_basis(indices, angle);

        let pops = self.make_precision_ops();
        let memory = self.make_memory_container(&pops);
        let substate = &self.substate;
        let measured = match measured {
            Some(measured) => measured,
            None => {
                let r: P = P::from(rand::random::<f64>()).unwrap() * self.substate.mag;
                let m = (0..1 << self.substate.n)
                    .try_fold(r, |r, i| {
                        let p = self
                            .substate
                            .rec_calculate_amplitude(i, &pops, DEFAULT_DEPTH, &memory)
                            .norm_sqr();
                        let r = r - p;
                        if r <= P::zero() {
                            Err(i)
                        } else {
                            Ok(r)
                        }
                    })
                    .err()
                    .unwrap();
                let indices: Vec<_> = indices
                    .iter()
                    .map(|indx| self.substate.n - 1 - *indx)
                    .collect();
                extract_bits(m, &indices)
            }
        };
        let p = measure_prob_fn(
            self.substate.n,
            measured,
            indices,
            Some(self.substate.input_offset),
            |index| substate.rec_calculate_amplitude(index, &pops, 0, &memory),
        );

        self.rotate_basis(indices, -angle);
        (measured, p)
    }

    fn state_magnitude(&self) -> P {
        self.substate.mag
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        self.rotate_basis(indices, angle);

        let pops = self.make_precision_ops();
        let memory = self.make_memory_container(&pops);
        let substate = &self.substate;
        let r = 0u64..1 << indices.len() as u64;
        let res = into_iter!(r)
            .map(|m| {
                measure_prob_fn(
                    substate.n,
                    m,
                    indices,
                    Some(substate.input_offset),
                    |index| substate.rec_calculate_amplitude(index, &pops, 0, &memory),
                )
            })
            .collect();
        self.rotate_basis(indices, -angle);
        res
    }

    fn into_state(self, order: Representation) -> Vec<Complex<P>> {
        match order {
            Representation::LittleEndian => (0..1 << self.substate.n)
                .map(|m| self.calculate_amplitude(flip_bits(self.substate.n as usize, m)))
                .collect(),
            Representation::BigEndian => (0..1 << self.substate.n)
                .map(|m| self.calculate_amplitude(m))
                .collect(),
        }
    }
}

#[cfg(test)]
mod feynmann_test {
    use super::*;
    use crate::state_ops::make_matrix_op;
    use crate::CircuitError;
    use num::One;

    fn lower_bits_state() -> Vec<Complex<f64>> {
        vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::zero(),
            Complex::zero(),
            Complex::zero(),
            Complex::zero(),
        ]
    }

    #[test]
    fn test_empty_initial_conditions() {
        let n = 3;
        let state = FeynmanState::<f64>::new(n);

        assert_eq!(state.calculate_amplitude(0), Complex::one());
        for i in 1..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
    }

    #[test]
    fn test_index_initial_conditions() {
        let n = 3;
        let init = [(vec![0, 1, 2], InitialState::Index(1))];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        assert_eq!(state.calculate_amplitude(0), Complex::zero());
        assert_eq!(state.calculate_amplitude(1), Complex::one());
        for i in 2..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }

        let init = [(vec![0, 1, 2], InitialState::Index(6))];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        for i in 0..6 {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
        assert_eq!(state.calculate_amplitude(6), Complex::one());
        for i in 7..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
    }

    #[test]
    fn test_product_index_initial_conditions() {
        let n = 3;
        let init = [
            (vec![1], InitialState::Index(1)),
            (vec![2], InitialState::Index(1)),
        ];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        for i in 0..3 {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
        assert_eq!(state.calculate_amplitude(3), Complex::one());
        for i in 4..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
    }

    #[test]
    fn test_fully_initial_conditions() {
        let n = 3;
        let init = [(vec![0, 1, 2], InitialState::FullState(lower_bits_state()))];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        for i in 0..4 {
            assert_eq!(state.calculate_amplitude(i), Complex::new(0.5, 0.0))
        }
        for i in 4..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
    }

    #[test]
    fn test_product_initial_conditions() {
        let n = 3;
        let init = [
            (
                vec![0],
                InitialState::FullState(vec![Complex::one(), Complex::zero()]),
            ),
            (
                vec![1],
                InitialState::FullState(vec![Complex::zero(), Complex::one()]),
            ),
            (
                vec![2],
                InitialState::FullState(vec![Complex::one(), Complex::zero()]),
            ),
        ];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        for i in 0..2 {
            assert_eq!(state.calculate_amplitude(i), Complex::zero());
        }
        assert_eq!(state.calculate_amplitude(2), Complex::one());
        for i in 3..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero());
        }
    }

    #[test]
    fn test_into_state() {
        let n = 3;
        let init_state = lower_bits_state();
        let init = [(vec![0, 1, 2], InitialState::FullState(init_state.clone()))];
        let state = FeynmanState::<f64>::new_from_initial_states(n, &init);
        assert_eq!(state.into_state(Representation::BigEndian), init_state)
    }

    #[test]
    fn test_stochastic_measure() {
        let n = 3;
        let init_state = lower_bits_state();
        let init = [(vec![0, 1, 2], InitialState::FullState(init_state.clone()))];
        let mut state = FeynmanState::<f64>::new_from_initial_states(n, &init);
        assert_eq!(
            state.stochastic_measure(&vec![1, 2], 0.0),
            vec![0.25, 0.25, 0.25, 0.25]
        )
    }

    #[test]
    fn test_bitflip_op_apply() -> Result<(), CircuitError> {
        let n = 3;
        let init = [(vec![0, 1, 2], InitialState::FullState(lower_bits_state()))];
        let mut state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        for i in 0..4 {
            assert_eq!(state.calculate_amplitude(i), Complex::new(0.5, 0.0))
        }
        for i in 4..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }

        // bitflip the largest bit
        let op = make_matrix_op(
            vec![0],
            vec![
                Complex::zero(),
                Complex::one(),
                Complex::one(),
                Complex::zero(),
            ],
        )?;
        state.apply_op(&op);

        for i in 0..4 {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
        for i in 4..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::new(0.5, 0.0))
        }
        Ok(())
    }

    #[test]
    fn test_measure_op_apply() {
        let n = 3;
        let init = [(vec![0, 1, 2], InitialState::FullState(lower_bits_state()))];
        let mut state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        let (m, p) = state.measure(&[0], None, 0.0);

        assert_eq!(m, 0);
        assert_eq!(p, 1.0);

        let (m, p) = state.measure(
            &[1],
            Some(MeasuredCondition {
                measured: 0,
                prob: None,
            }),
            0.0,
        );

        assert_eq!(m, 0);
        assert_eq!(p, 0.5);
    }

    #[test]
    fn test_measure_effect() {
        let n = 3;
        let init = [(vec![0, 1, 2], InitialState::FullState(lower_bits_state()))];
        let mut state = FeynmanState::<f64>::new_from_initial_states(n, &init);

        let (m, p) = state.measure(
            &[1],
            Some(MeasuredCondition {
                measured: 0,
                prob: None,
            }),
            0.0,
        );

        assert_eq!(m, 0);
        assert_eq!(p, 0.5);

        let c = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        for i in 0..2 {
            // Off by ~1e16
            assert!((state.calculate_amplitude(i) - c).norm() < 1e-10);
        }
        for i in 3..1 << n {
            assert_eq!(state.calculate_amplitude(i), Complex::zero())
        }
    }
}
