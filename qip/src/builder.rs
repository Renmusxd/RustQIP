use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::ops::Neg;

use num_rational::{Ratio, Rational64};
use num_traits::{One, ToPrimitive, Zero};
use qip_iterators::matrix_ops::apply_op_overwrite;

use crate::builder_traits::*;
use crate::conditioning::{Conditionable, ConditionableSubcircuit};
use crate::errors::{CircuitError, CircuitResult};
use crate::inverter::Invertable;
use crate::inverter::RecursiveCircuitBuilder;
use crate::state_ops::matrix_ops::{make_control_op, make_matrix_op, make_swap_op};
use crate::state_ops::measurement_ops::{measure, measure_probs};
use crate::types::Precision;
use crate::Complex;

/// A local circuit builder for constructing circuits out of standard gates.
/// LocalBuilder breaks complicated multi-register gates, like toffoli, into combinations of simple
/// gates like CNOT.
#[derive(Default, Debug)]
pub struct LocalBuilder<P: Precision> {
    pipeline: Vec<(Vec<usize>, BuilderCircuitObject<P>)>,
    n: usize,
    zeroed_qubits: Vec<Qudit>,
    measurements: usize,
}

impl<P: Precision> LocalBuilder<P> {
    /// Applies a phase to a qubit. This has no effect by default but has an impact if conditioned.
    pub fn apply_global_phase(&mut self, r: Qudit, theta: P) -> Qudit {
        let co = BuilderCircuitObject {
            n: r.n(),
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::GlobalPhase(
                RotationObject::Floating(theta),
            )),
        };
        self.apply_circuit_object(r, co).unwrap()
    }

    /// Applied a global phase to a qubit, expressed as a ratio rather than a float.
    pub fn apply_global_phase_ratio(&mut self, r: Qudit, theta: Rational64) -> Qudit {
        let co = BuilderCircuitObject {
            n: r.n(),
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::GlobalPhase(
                RotationObject::PiRational(theta),
            )),
        };
        self.apply_circuit_object(r, co).unwrap()
    }

    /// Apply a global phase of pi/m for integer m.
    pub fn apply_global_phase_pi_by(&mut self, r: Qudit, m: i64) -> Qudit {
        self.apply_global_phase_ratio(r, Ratio::new(1, m))
    }

    /// Returns the depth of the current circuit (pipeline).
    pub fn pipeline_depth(&self) -> usize {
        self.pipeline.len()
    }
}

/// The register implementation for the LocalBuilder.
#[derive(Debug)]
pub struct Qudit {
    indices: Vec<usize>,
}

impl QubitRegister for Qudit {
    fn n(&self) -> usize {
        self.indices.len()
    }

    fn indices(&self) -> &[usize] {
        self.indices.as_ref()
    }
}

impl Qudit {
    fn new<It>(indices: It) -> Option<Self>
    where
        It: Into<Vec<usize>>,
    {
        let indices = indices.into();
        if !indices.is_empty() {
            Some(Self { indices })
        } else {
            None
        }
    }
    fn new_from_iter<It>(indices: It) -> Option<Self>
    where
        It: Iterator<Item = usize>,
    {
        let indices = indices.into_iter().collect::<Vec<_>>();
        Self::new(indices)
    }
}

/// A pipeline object for the LocalBuilder.
#[derive(Debug, Clone)]
pub struct BuilderCircuitObject<P: Precision> {
    n: usize,
    object: BuilderCircuitObjectType<P>,
}

/// The type of pipeline object for LocalBuilder.
#[derive(Debug, Clone)]
pub enum BuilderCircuitObjectType<P: Precision> {
    /// A unitary operation on the circuit.
    Unitary(UnitaryMatrixObject<P>),
    /// A measurement operation on the circuit.
    Measurement(MeasurementObject),
}

/// The type of unitary matrix for LocalBuilder.
#[derive(Debug, Clone)]
pub enum UnitaryMatrixObject<P: Precision> {
    /// A pauli X gate.
    X,
    /// A pauli Y gate.
    Y,
    /// A pauli Z gate.
    Z,
    /// A hadamard gate.
    H,
    /// A conditional phase gate by pi/4.
    S,
    /// A conditional phase gate by pi/8.
    T,
    /// A controlled pauli X gate, first qubit is control.
    CNOT,
    /// A swap gate between two qubits.
    SWAP,
    /// A conditional phase gate by RotationObject phase.
    Rz(RotationObject<P>),
    /// A Generic Matrix
    MAT(Vec<Complex<P>>),
    /// A global phase applied on this qubit.
    /// This normally doesn't matter but can have an effect
    /// when we are conditioning them.
    GlobalPhase(RotationObject<P>),
}

/// Represents phase floats.
#[derive(Debug, Clone)]
pub enum RotationObject<P: Precision> {
    /// A rotation represented by a floating point precision.
    Floating(P),
    /// A rotation represented by a fixed ratio times pi.
    PiRational(Ratio<i64>),
}

impl<P: Precision> Neg for RotationObject<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            RotationObject::Floating(f) => Self::Floating(-f),
            RotationObject::PiRational(r) => Self::PiRational(-r),
        }
    }
}

impl<P: Precision> PartialEq for UnitaryMatrixObject<P> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::X, Self::X)
            | (Self::Y, Self::Y)
            | (Self::Z, Self::Z)
            | (Self::H, Self::H)
            | (Self::S, Self::S)
            | (Self::T, Self::T)
            | (Self::CNOT, Self::CNOT)
            | (Self::SWAP, Self::SWAP) => true,
            (Self::Rz(ra), Self::Rz(rb)) => ra.eq(rb),
            (Self::MAT(ma), Self::MAT(mb)) => ma.eq(mb),
            (Self::GlobalPhase(ra), Self::GlobalPhase(rb)) => ra.eq(rb),
            (_, _) => false,
        }
    }
}

impl<P: Precision> PartialEq for BuilderCircuitObjectType<P> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unitary(ua), Self::Unitary(ub)) => ua.eq(ub),
            (Self::Measurement(ma), Self::Measurement(mb)) => ma.eq(mb),
            (_, _) => false,
        }
    }
}

impl<P: Precision> PartialEq for RotationObject<P> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Floating(pa), Self::Floating(pb)) => pa.eq(pb),
            (Self::PiRational(ra), Self::PiRational(rb)) => ra.eq(rb),
            (_, _) => false,
        }
    }
}

impl<P: Precision> Eq for UnitaryMatrixObject<P> {}

impl<P: Precision> Eq for BuilderCircuitObjectType<P> {}

impl<P: Precision> Eq for RotationObject<P> {}

fn hash_p<P: Precision, H: Hasher>(f: P, state: &mut H) {
    format!("{}", f).hash(state)
}

impl<P: Precision> Hash for BuilderCircuitObjectType<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            BuilderCircuitObjectType::Measurement(m) => {
                state.write_i8(0);
                m.hash(state)
            }
            BuilderCircuitObjectType::Unitary(u) => {
                state.write_i8(1);
                u.hash(state)
            }
        }
    }
}

impl<P: Precision> Hash for UnitaryMatrixObject<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            UnitaryMatrixObject::X => state.write_i8(0),
            UnitaryMatrixObject::Y => state.write_i8(1),
            UnitaryMatrixObject::Z => state.write_i8(2),
            UnitaryMatrixObject::H => state.write_i8(3),
            UnitaryMatrixObject::S => state.write_i8(4),
            UnitaryMatrixObject::T => state.write_i8(5),
            UnitaryMatrixObject::CNOT => state.write_i8(6),
            UnitaryMatrixObject::SWAP => state.write_i8(7),
            UnitaryMatrixObject::Rz(rot) => {
                state.write_i8(8);
                rot.hash(state);
            }
            UnitaryMatrixObject::GlobalPhase(rot) => {
                state.write_i8(9);
                rot.hash(state);
            }
            UnitaryMatrixObject::MAT(data) => {
                state.write_i8(10);
                data.iter().for_each(|c| {
                    hash_p(c.re, state);
                    hash_p(c.im, state);
                })
            }
        }
    }
}

impl<P: Precision> Hash for RotationObject<P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            // Grossly inefficient but also don't hash floats.
            RotationObject::Floating(f) => hash_p(*f, state),
            RotationObject::PiRational(r) => r.hash(state),
        }
    }
}

/// Represents a type of measurement in the circuit.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum MeasurementObject {
    /// Performs a single measurement and collapses the wavefunction.
    Measurement,
    /// Simulates a series of measurements and returns a probability distribution over results.
    /// Does not collapse the wavefunction.
    StochasticMeasurement,
}

/// Represents the result of a measurement on the circuit.
#[derive(Debug, Clone)]
pub enum MeasurementResults<P: Precision> {
    /// The result of a single measurement on selected qubits, returns the measurement as well as
    /// the likelyhood of that measurement taking place.
    Single(usize, P),
    /// The probability of each measurement indexed by the measurement itself.
    Stochastic(Vec<P>),
}

/// A series of measurement results at the end of the circuit.
#[derive(Debug)]
pub struct Measurements<P: Precision> {
    measurements: Vec<MeasurementResults<P>>,
}

impl<P: Precision> Measurements<P> {
    /// Get a measurement result given a handle.
    pub fn get_measurement(&self, handle: MeasurementHandle) -> (usize, P) {
        match &self.measurements[handle.id] {
            MeasurementResults::Single(val, prob) => (*val, *prob),
            MeasurementResults::Stochastic(_) => unreachable!(),
        }
    }
    /// Get a stochastic measurement result given a handle.
    pub fn get_stochastic_measurement(&self, handle: StochasticMeasurementHandle) -> &[P] {
        match &self.measurements[handle.id] {
            MeasurementResults::Single(_, _) => unreachable!(),
            MeasurementResults::Stochastic(probs) => probs.as_slice(),
        }
    }
}

impl<P: Precision> CircuitBuilder for LocalBuilder<P> {
    type Register = Qudit;
    type CircuitObject = BuilderCircuitObject<P>;
    type StateCalculation = (Vec<Complex<P>>, Measurements<P>);

    fn n(&self) -> usize {
        self.n
    }

    fn register(&mut self, n: NonZeroUsize) -> Self::Register {
        let n: usize = n.into();
        let r = Self::Register::new_from_iter(self.n..self.n + n).unwrap();
        self.n += n;
        r
    }

    fn merge_two_registers(&mut self, r1: Self::Register, r2: Self::Register) -> Self::Register {
        Self::Register::new_from_iter(r1.indices.into_iter().chain(r2.indices)).unwrap()
    }

    fn split_register_relative<It>(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitResult<Self::Register>
    where
        It: IntoIterator<Item = usize>,
    {
        let selected_indices = indices.into_iter().filter_map(|i| {
            if i <= r.indices.len() {
                Some(r.indices[i])
            } else {
                None
            }
        });
        let r1 = Self::Register::new_from_iter(selected_indices);

        let remaining_indices = r.indices.into_iter().filter(|oi| match &r1 {
            Some(r1) => !r1.indices.contains(oi),
            None => true,
        });
        let r2 = Self::Register::new_from_iter(remaining_indices);

        match (r1, r2) {
            (Some(r1), None) => SplitResult::SELECTED(r1),
            (None, Some(r2)) => SplitResult::UNSELECTED(r2),
            (Some(r1), Some(r2)) => SplitResult::SPLIT(r1, r2),
            (None, None) => unreachable!(),
        }
    }

    fn apply_circuit_object(
        &mut self,
        r: Self::Register,
        c: Self::CircuitObject,
    ) -> CircuitResult<Self::Register> {
        if c.n == 1 || c.n == r.n() {
            if c.n == 1 && r.n() > 1 {
                // Do broadcasting
                let rs = self.split_all_register(r);
                rs.iter()
                    .for_each(|r| self.pipeline.push((r.indices.clone(), c.clone())));
                Ok(self.merge_registers(rs).unwrap())
            } else {
                // Normal application.
                self.pipeline.push((r.indices.clone(), c));
                Ok(r)
            }
        } else {
            Err(CircuitError::new(
                "Matrix has incorrect N and cannot be broadcast",
            ))
        }
    }

    fn calculate_state_with_init<'a, It>(&mut self, it: It) -> Self::StateCalculation
    where
        Self::Register: 'a,
        It: IntoIterator<Item = (&'a Self::Register, usize)>,
    {
        let n = self.n();
        let mut state = vec![Complex::zero(); 1 << n];
        let arena = state.clone();

        let mut initial_index = 0;
        it.into_iter()
            .flat_map(|(r, x)| {
                let rn = r.n();
                r.indices
                    .iter()
                    .rev()
                    .cloned()
                    .enumerate()
                    .map(move |(ri, i)| (n - 1 - i, (x >> (rn - 1 - ri)) & 1))
            })
            .for_each(|(index, bit)| initial_index |= bit << index);
        state[initial_index] = Complex::one();

        let (state, _, measurements) = self
            .pipeline
            .iter()
            .try_fold(
                (state, arena, vec![]),
                |(state, mut arena, mut measurements), (indices, obj)| -> CircuitResult<_> {
                    let BuilderCircuitObject { object, .. } = obj;
                    match object {
                        // Global phases do not affect state.
                        BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::GlobalPhase(_)) => {}
                        // Other unitaries do though.
                        BuilderCircuitObjectType::Unitary(object) => {
                            let indices = indices.clone();
                            let l = Complex::one();
                            let o = Complex::zero();
                            let i = Complex::i();
                            let uop = match object {
                                UnitaryMatrixObject::X => make_matrix_op(indices, vec![o, l, l, o]),
                                UnitaryMatrixObject::Y => {
                                    make_matrix_op(indices, vec![o, -i, i, o])
                                }
                                UnitaryMatrixObject::Z => {
                                    make_matrix_op(indices, vec![l, o, o, -l])
                                }
                                UnitaryMatrixObject::H => {
                                    let nl = Complex::one()
                                        * P::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
                                    make_matrix_op(indices, vec![nl, nl, nl, -nl])
                                }
                                UnitaryMatrixObject::S => make_matrix_op(indices, vec![l, o, o, i]),
                                UnitaryMatrixObject::T => {
                                    let t = Complex::from_polar(
                                        P::one(),
                                        P::from(std::f64::consts::FRAC_PI_4).unwrap(),
                                    );
                                    make_matrix_op(indices, vec![l, o, o, t])
                                }
                                UnitaryMatrixObject::CNOT => {
                                    let cindex = vec![indices[0]];
                                    let indices = indices[1..].to_vec();
                                    make_control_op(
                                        cindex,
                                        make_matrix_op(indices, vec![o, l, l, o])?,
                                    )
                                }
                                UnitaryMatrixObject::MAT(data) => {
                                    make_matrix_op(indices, data.clone())
                                }
                                UnitaryMatrixObject::SWAP => {
                                    let n = indices.len();
                                    assert_eq!(n % 2, 0);
                                    let x = n / 2;
                                    let a_indices = indices[..x].to_vec();
                                    let b_indices = indices[x..].to_vec();
                                    make_swap_op(a_indices, b_indices)
                                }
                                UnitaryMatrixObject::Rz(theta) => {
                                    let theta = match theta {
                                        RotationObject::Floating(p) => *p,
                                        RotationObject::PiRational(r) => {
                                            r.to_f64().and_then(|f| P::from(f)).unwrap()
                                        }
                                    };
                                    let h_theta = theta * P::from(0.5).unwrap();
                                    make_matrix_op(
                                        indices,
                                        vec![
                                            Complex::from_polar(P::one(), -h_theta),
                                            Complex::zero(),
                                            Complex::zero(),
                                            Complex::from_polar(P::one(), h_theta),
                                        ],
                                    )
                                }
                                UnitaryMatrixObject::GlobalPhase(_) => unreachable!(),
                            }?;
                            apply_op_overwrite(n, &uop, &state, &mut arena, 0, 0);
                        }
                        BuilderCircuitObjectType::Measurement(object) => match object {
                            MeasurementObject::Measurement => {
                                let (measured, p) =
                                    measure(n, indices, &state, &mut arena, None, None);
                                measurements.push(MeasurementResults::Single(measured, p));
                            }
                            MeasurementObject::StochasticMeasurement => {
                                let ps = measure_probs(n, indices, &state, None);
                                measurements.push(MeasurementResults::Stochastic(ps));
                            }
                        },
                    }

                    Ok((arena, state, measurements))
                },
            )
            .unwrap();
        (state, Measurements { measurements })
    }
}

impl<P: Precision> UnitaryBuilder<P> for LocalBuilder<P> {
    fn vec_matrix_to_circuitobject(n: usize, data: Vec<Complex<P>>) -> Self::CircuitObject {
        Self::CircuitObject {
            n,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::MAT(data)),
        }
    }
}

impl<P: Precision> CliffordTBuilder<P> for LocalBuilder<P> {
    fn make_x(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::X),
        }
    }
    fn make_y(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::Y),
        }
    }
    fn make_z(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::Z),
        }
    }
    fn make_h(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::H),
        }
    }
    fn make_s(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::S),
        }
    }
    fn make_t(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 1,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::T),
        }
    }
    fn make_cnot(&self) -> Self::CircuitObject {
        Self::CircuitObject {
            n: 2,
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::CNOT),
        }
    }
}

impl<P: Precision> TemporaryRegisterBuilder for LocalBuilder<P> {
    fn make_zeroed_temp_qubit(&mut self) -> Self::Register {
        if let Some(r) = self.zeroed_qubits.pop() {
            r
        } else {
            self.qubit()
        }
    }

    fn return_zeroed_temp_register(&mut self, r: Self::Register) {
        let rs = self.split_all_register(r);
        self.zeroed_qubits.extend(rs);
    }
}

impl<P: Precision> AdvancedCircuitBuilder<P> for LocalBuilder<P> {}

/// A handle which points to a measurement result.
#[derive(Debug, Clone, Copy)]
pub struct MeasurementHandle {
    id: usize,
}

impl<P: Precision> MeasurementBuilder for LocalBuilder<P> {
    type MeasurementHandle = MeasurementHandle;

    fn measure(&mut self, r: Self::Register) -> (Self::Register, Self::MeasurementHandle) {
        let obj = BuilderCircuitObject {
            n: r.n(),
            object: BuilderCircuitObjectType::Measurement(MeasurementObject::Measurement),
        };
        self.pipeline.push((r.indices.clone(), obj));
        let m = self.measurements;
        self.measurements += 1;
        (r, Self::MeasurementHandle { id: m })
    }
}

/// A handle which points to a stochastic measurement result.
#[derive(Debug, Clone, Copy)]
pub struct StochasticMeasurementHandle {
    id: usize,
}

impl<P: Precision> StochasticMeasurementBuilder for LocalBuilder<P> {
    type StochasticMeasurementHandle = StochasticMeasurementHandle;

    fn measure_stochastic(
        &mut self,
        r: Self::Register,
    ) -> (Self::Register, Self::StochasticMeasurementHandle) {
        let obj = BuilderCircuitObject {
            n: r.n(),
            object: BuilderCircuitObjectType::Measurement(MeasurementObject::StochasticMeasurement),
        };
        self.pipeline.push((r.indices.clone(), obj));
        let m = self.measurements;
        self.measurements += 1;
        (r, Self::StochasticMeasurementHandle { id: m })
    }
}

impl<P: Precision> RotationsBuilder<P> for LocalBuilder<P> {
    fn rz(&mut self, r: Self::Register, theta: P) -> Self::Register {
        let co = Self::CircuitObject {
            n: r.n(),
            object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::Rz(
                RotationObject::Floating(theta),
            )),
        };
        self.apply_circuit_object(r, co).unwrap()
    }
    fn rz_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        if m == 0 {
            Err(CircuitError::new("Cannot rotate by pi/0"))
        } else {
            let co = Self::CircuitObject {
                n: r.n(),
                object: BuilderCircuitObjectType::Unitary(UnitaryMatrixObject::Rz(
                    RotationObject::PiRational(Ratio::new(1, m)),
                )),
            };
            self.apply_circuit_object(r, co)
        }
    }
}

impl<P: Precision> Conditionable for LocalBuilder<P> {
    fn try_apply_with_condition(
        &mut self,
        cr: Self::Register,
        r: Self::Register,
        co: Self::CircuitObject,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        match co.object {
            BuilderCircuitObjectType::Unitary(unit) => match unit {
                UnitaryMatrixObject::X => self.toffoli(cr, r),
                UnitaryMatrixObject::Y => {
                    let r = self.s(r);
                    let (cr, r) = self.toffoli(cr, r)?;
                    let r = self.s_dagger(r);
                    Ok((cr, r))
                }
                UnitaryMatrixObject::Z => {
                    let r = self.h(r);
                    let (cr, r) = self.toffoli(cr, r)?;
                    let r = self.h(r);
                    Ok((cr, r))
                }
                UnitaryMatrixObject::H => {
                    let r = self.ry_pi_by(r, 4)?;
                    let (cr, r) = self.toffoli(cr, r)?;
                    let r = self.ry_pi_by(r, -4)?;
                    Ok((cr, r))
                }
                UnitaryMatrixObject::S => {
                    let cr = self.merge_two_registers(cr, r);
                    let tq = self.make_zeroed_temp_qubit();
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    let tq = self.s(tq);
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    self.return_zeroed_temp_register(tq);
                    let (cr, r) = self.split_last_qubit(cr);
                    let r = r.unwrap();
                    Ok((cr, r))
                }
                UnitaryMatrixObject::T => {
                    let cr = self.merge_two_registers(cr, r);
                    let tq = self.make_zeroed_temp_qubit();
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    let tq = self.t(tq);
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    self.return_zeroed_temp_register(tq);
                    let (cr, r) = self.split_last_qubit(cr);
                    let r = r.unwrap();
                    Ok((cr, r))
                }
                UnitaryMatrixObject::SWAP => {
                    let n = r.n();
                    assert_eq!(n % 2, 0);
                    let rs = self.split_all_register(r);
                    let (ra, rb) = split_vector_at(rs, n / 2);
                    let (cr, ras, rbs) = ra.into_iter().zip(rb.into_iter()).try_fold(
                        (cr, vec![], vec![]),
                        |(cr, mut ras, mut rbs), (ra, rb)| {
                            debug_assert_eq!(ra.n(), 1);
                            debug_assert_eq!(rb.n(), 1);

                            // With the program macro we would be doing this:
                            // let (cr, ra, rb) = program!(self, cr, ra, rb;
                            //     control not |cr, ra,| rb;
                            //     control not |cr, rb,| ra;
                            //     control not |cr, ra,| rb;
                            // )?;

                            let new_cr = self.merge_two_registers(cr, ra);
                            let (new_cr, rb) = self.cnot(new_cr, rb)?;
                            let (cr, ra) = self.split_last_qubit(new_cr);
                            let ra = ra.unwrap();

                            let new_cr = self.merge_two_registers(cr, rb);
                            let (new_cr, ra) = self.cnot(new_cr, ra)?;
                            let (cr, rb) = self.split_last_qubit(new_cr);
                            let rb = rb.unwrap();

                            let new_cr = self.merge_two_registers(cr, ra);
                            let (new_cr, rb) = self.cnot(new_cr, rb)?;
                            let (cr, ra) = self.split_last_qubit(new_cr);
                            let ra = ra.unwrap();

                            ras.push(ra);
                            rbs.push(rb);
                            Ok((cr, ras, rbs))
                        },
                    )?;
                    let r = self.merge_registers(ras.into_iter().chain(rbs)).unwrap();
                    Ok((cr, r))
                }
                UnitaryMatrixObject::CNOT => {
                    assert_eq!(r.n(), 2);
                    let (ra, r) = self.split_first_qubit(r);
                    let ra = ra.unwrap();
                    let cr = self.merge_two_registers(cr, ra);
                    let (cr, r) = self.toffoli(cr, r)?;
                    let (cr, ra) = self.split_last_qubit(cr);
                    let ra = ra.unwrap();
                    let r = self.merge_two_registers(ra, r);
                    Ok((cr, r))
                }
                UnitaryMatrixObject::GlobalPhase(phase) => {
                    // So a global phase applied to r if cr is actually just a phase gate on the
                    // conditioned qubits, r doesn't enter.

                    let tq = self.make_zeroed_temp_qubit();
                    let (cr, tq) = self.toffoli(cr, tq)?;

                    let tq = match phase {
                        RotationObject::Floating(theta) => {
                            let half_phase = theta * P::from(0.5).unwrap();
                            let tq = self.rz(tq, half_phase);
                            self.apply_global_phase(tq, half_phase)
                        }
                        RotationObject::PiRational(r) => {
                            let half_phase = r / 2;
                            let tq = self.rz_ratio(tq, half_phase)?;
                            self.apply_global_phase_ratio(tq, half_phase)
                        }
                    };

                    let (cr, tq) = self.toffoli(cr, tq)?;
                    self.return_zeroed_temp_register(tq);
                    Ok((cr, r))
                }
                UnitaryMatrixObject::Rz(phase) => {
                    let crn = cr.n();
                    let cr = self.merge_two_registers(cr, r);

                    let tq = self.make_zeroed_temp_qubit();
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    let tq = match phase {
                        RotationObject::Floating(phase) => self.rz(tq, phase),
                        RotationObject::PiRational(phase) => self.rz_ratio(tq, phase)?,
                    };
                    let (cr, tq) = self.toffoli(cr, tq)?;
                    self.return_zeroed_temp_register(tq);

                    match self.split_register_relative(cr, 0..crn) {
                        SplitResult::SPLIT(cr, r) => Ok((cr, r)),
                        SplitResult::SELECTED(_) => unreachable!(),
                        SplitResult::UNSELECTED(_) => unreachable!(),
                    }
                }
                UnitaryMatrixObject::MAT(_data) => todo!(),
            },
            BuilderCircuitObjectType::Measurement(_) => {
                Err(CircuitError::new("Cannot condition measurements."))
            }
        }
    }
}

fn split_vector_at<T>(mut v: Vec<T>, x: usize) -> (Vec<T>, Vec<T>) {
    let n = v.len();
    assert!(n >= x);
    let mut b = vec![];
    for _ in 0..(n - x) {
        b.push(v.pop().unwrap());
    }
    b.reverse();
    (v, b)
}

impl<P: Precision> Subcircuitable for LocalBuilder<P> {
    type Subcircuit = Vec<(Vec<usize>, Self::CircuitObject)>;

    fn make_subcircuit(&self) -> CircuitResult<Self::Subcircuit> {
        Ok(self.pipeline.clone())
    }

    fn apply_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        r: Self::Register,
    ) -> CircuitResult<Self::Register> {
        apply_pipeline_objects(self, sc, r)
    }
}

impl<P: Precision> Invertable for LocalBuilder<P> {
    type SimilarBuilder = Self;

    fn new_similar(&self) -> Self {
        Self::default()
    }

    fn invert_subcircuit(sc: Self::Subcircuit) -> CircuitResult<Self::Subcircuit> {
        sc.into_iter()
            .rev()
            .try_fold(vec![], |mut acc, (indices, co)| {
                invert_circuit_object(co)?
                    .into_iter()
                    .for_each(|co| acc.push((indices.clone(), co)));
                Ok(acc)
            })
    }
}

impl<P: Precision> ConditionableSubcircuit for LocalBuilder<P> {
    fn apply_conditioned_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        let mut cb = self.condition_with(cr);
        let r = apply_pipeline_objects(&mut cb, sc, r)?;
        let cr = cb.dissolve();
        Ok((cr, r))
    }
}

fn apply_pipeline_objects<CB, CO>(
    cb: &mut CB,
    sc: CB::Subcircuit,
    r: CB::Register,
) -> CircuitResult<CB::Register>
where
    CB: CircuitBuilder<CircuitObject = CO>
        + Subcircuitable<Subcircuit = Vec<(Vec<usize>, CO)>>
        + TemporaryRegisterBuilder,
{
    let rn = r.n();
    let mut rs = cb.split_all_register(r);
    let max_r_index = sc
        .iter()
        .flat_map(|(indices, _)| indices.iter().cloned().max())
        .max()
        .unwrap();
    // Need temp qubits for excess.
    if let Some(temp_n) = NonZeroUsize::new(1 + max_r_index - rn) {
        let temp = cb.make_zeroed_temp_register(temp_n);
        rs.extend(cb.split_all_register(temp));
    }
    let mut rs = rs.into_iter().map(Some).collect::<Vec<_>>();
    sc.into_iter().try_for_each(|(indices, co)| {
        let sub_rs = indices.iter().map(|index| rs[*index].take().unwrap());
        let sub_r = cb.merge_registers(sub_rs).unwrap();
        let sub_r = cb.apply_circuit_object(sub_r, co)?;
        let sub_rs = cb.split_all_register(sub_r);
        indices
            .into_iter()
            .zip(sub_rs.into_iter())
            .for_each(|(index, r)| {
                rs[index] = Some(r);
            });
        assert!(rs.iter().all(|r| r.is_some()), "Not all qubits returned");
        Ok(())
    })?;
    let rs = rs.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    let (rs, trs) = split_vector_at(rs, rn);
    if let Some(tr) = cb.merge_registers(trs) {
        cb.return_zeroed_temp_register(tr);
    }
    let r = cb.merge_registers(rs).unwrap();
    Ok(r)
}

fn invert_circuit_object<P: Precision>(
    co: BuilderCircuitObject<P>,
) -> CircuitResult<Vec<BuilderCircuitObject<P>>> {
    match co.object {
        BuilderCircuitObjectType::Unitary(u) => {
            let new_objs = match u {
                UnitaryMatrixObject::X
                | UnitaryMatrixObject::Y
                | UnitaryMatrixObject::Z
                | UnitaryMatrixObject::H
                | UnitaryMatrixObject::CNOT
                | UnitaryMatrixObject::SWAP => vec![u],
                UnitaryMatrixObject::S => vec![UnitaryMatrixObject::Z, u],
                UnitaryMatrixObject::T => vec![UnitaryMatrixObject::Z, UnitaryMatrixObject::S, u],
                UnitaryMatrixObject::Rz(phase) => vec![UnitaryMatrixObject::Rz(phase.neg())],
                UnitaryMatrixObject::GlobalPhase(phase) => {
                    vec![UnitaryMatrixObject::GlobalPhase(phase.neg())]
                }
                UnitaryMatrixObject::MAT(data) => {
                    // Matrix is unitary, inverse is just dagger
                    let mut inverse_data = vec![Complex::zero(); data.len()];
                    let nn = 1 << co.n;
                    (0..nn).for_each(|i| {
                        (0..nn).for_each(|j| {
                            let index_a = i * nn + j;
                            let index_b = j * nn + i;
                            inverse_data[index_a] = data[index_b].conj();
                        })
                    });
                    vec![UnitaryMatrixObject::MAT(inverse_data)]
                }
            };
            Ok(new_objs
                .into_iter()
                .map(|new_obj| BuilderCircuitObject {
                    n: co.n,
                    object: BuilderCircuitObjectType::Unitary(new_obj),
                })
                .collect())
        }
        BuilderCircuitObjectType::Measurement(_) => {
            Err(CircuitError::new("Cannot invert measurement."))
        }
    }
}

impl<P: Precision> RecursiveCircuitBuilder<P> for LocalBuilder<P> {
    type RecursiveSimilarBuilder = Self::SimilarBuilder;
}
