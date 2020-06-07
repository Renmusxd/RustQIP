use crate::iterators::{
    par_sum_from_iterator, precision_get_index, precision_num_indices,
    sum_for_op_cols, PrecisionUnitaryOp,
};
use crate::measurement_ops::{measure_prob_fn, MeasuredCondition};
use crate::pipeline::{InitialState, Representation};
use crate::rayon_helper::*;
use crate::state_ops::{clone_as_precision_op, full_to_sub, sub_to_full, UnitaryOp};
use crate::utils::flip_bits;
use crate::{Precision, QuantumState};
use num::{Complex, Zero};

/// TODO
#[derive(Debug)]
pub struct FeynmanState<P: Precision> {
    ops: Vec<FeynmanOp<P>>,
    substate: FeynmanThreadSafeState<P>,
}

#[derive(Debug)]
struct FeynmanThreadSafeState<P: Precision> {
    n: u64,
    mag: P,
    input_offset: u64,
    output_offset: u64,
    initial_state: Vec<(Vec<u64>, InitialState<P>)>,
}

#[derive(Debug)]
enum FeynmanOp<P: Precision> {
    OP(UnitaryOp),
    MEASUREMENT(u64, Vec<u64>, P),
}

#[derive(Debug)]
enum FeynmanPrecisionOp<'a, P: Precision> {
    OP(PrecisionUnitaryOp<'a, P>, Vec<u64>),
    MEASUREMENT(u64, Vec<u64>, P),
}

impl<P: Precision> FeynmanState<P> {
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

    fn calculate_amplitude(&self, m: u64) -> Complex<P> {
        let pops = self.make_precision_ops();
        self.substate.rec_calculate_amplitude(m, &pops, true)
    }
}

impl<P: Precision> FeynmanThreadSafeState<P> {
    fn rec_calculate_amplitude(
        &self,
        m: u64,
        ops: &[FeynmanPrecisionOp<P>],
        allowed_parallel: bool,
    ) -> Complex<P> {
        match ops {
            [] => {
                // Get values from initial state
                self.initial_state
                    .iter()
                    .map(|(indices, init)| {
                        let subindex = full_to_sub(self.n, indices, m);
                        init.get_amplitude(subindex)
                    })
                    .product()
            }
            slice => {
                let head = &slice[0..slice.len() - 1];
                match &slice[slice.len() - 1] {
                    FeynmanPrecisionOp::OP(op, mat_indices) => {
                        // Maps from a op matrix column (from 0 to 2^nindices) to the value at that column
                        // for the row calculated above.
                        let f = |(i, val): (u64, Complex<P>)| -> Complex<P> {
                            let colbits = sub_to_full(self.n, mat_indices, i, m);
                            val * self.rec_calculate_amplitude(colbits, head, false)
                        };

                        let nindices = mat_indices.len() as u64;
                        let matrow = full_to_sub(self.n, mat_indices, m);

                        // We only want to allow parallelism for the top level.
                        if allowed_parallel {
                            par_sum_from_iterator(nindices, matrow, op, f)
                        } else {
                            sum_for_op_cols(nindices, matrow, op, f)
                        }
                    }
                    FeynmanPrecisionOp::MEASUREMENT(measured, indices, p) => {
                        if *measured == full_to_sub(self.n, indices, m) {
                            self.rec_calculate_amplitude(m, head, false) / p.sqrt()
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
        let mag = states.iter().map(|(_, s)| s.get_magnitude()).product();
        let substate = FeynmanThreadSafeState {
            n,
            mag,
            input_offset: 0,
            output_offset: 0,
            initial_state: states.to_vec(),
        };
        Self {
            ops: vec![],
            substate,
        }
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
        // TODO add angle usage
        if angle != 0.0 {
            unimplemented!()
        };

        match measured {
            Some(measured) => {
                let pops = self.make_precision_ops();
                let substate = &self.substate;
                let p = measure_prob_fn(
                    self.substate.n,
                    measured,
                    indices,
                    Some(self.substate.input_offset),
                    |index| substate.rec_calculate_amplitude(index, &pops, true)
                );
                (measured, p)
            },
            None => {
                let r: P = P::from(rand::random::<f64>()).unwrap() * self.substate.mag;
                (0..1 << self.substate.n)
                    .try_fold(r, |r, i| {
                        let p = self.calculate_amplitude(i).norm_sqr();
                        let r = r - p;
                        if r <= P::zero() {
                            Err((i, p))
                        } else {
                            Ok(r)
                        }
                    })
                    .err()
                    .unwrap()
            }
        }
    }

    fn state_magnitude(&self) -> P {
        self.substate.mag
    }

    fn stochastic_measure(&mut self, indices: &[u64], angle: f64) -> Vec<P> {
        // TODO add angle usage
        if angle != 0.0 {
            unimplemented!()
        };

        let pops = self.make_precision_ops();
        let substate = &self.substate;
        let r = 0u64 .. 1 << indices.len() as u64;
        into_iter!(r).map(|m| {
            measure_prob_fn(
                substate.n,
                m,
                indices,
                Some(substate.input_offset),
                |index| substate.rec_calculate_amplitude(index, &pops, true)
            )
        }).collect()
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
