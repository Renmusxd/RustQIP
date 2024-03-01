use crate::errors::{CircuitError, CircuitResult};
use crate::types::Precision;
use num_complex::Complex;
use num_rational::Rational64;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::num::NonZeroUsize;

/// Standard functions needed by registers containing multiple qubits.
pub trait QubitRegister {
    /// Size of the register in qubits.
    fn n(&self) -> usize;
    /// Size of the register in qubits.
    fn n_nonzero(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.n()).unwrap()
    }
    /// Absolute indices represented by the register.
    fn indices(&self) -> &[usize];
}

/// Result of splitting a register in two.
#[derive(Debug)]
pub enum SplitResult<R: QubitRegister + Debug> {
    /// All registers were selected
    SELECTED(R),
    /// None of the registers were selected
    UNSELECTED(R),
    /// Some registers were selected, some were not selected.
    SPLIT(R, R),
}

/// Result of splitting a register into multiple registers.
#[derive(Debug)]
pub enum SplitManyResult<R: QubitRegister + Debug> {
    /// All registers were selected.
    AllSelected(Vec<R>),
    /// Some were selected, remaining were not.
    Remaining(Vec<R>, R),
}

impl<R: QubitRegister + Debug> SplitManyResult<R> {
    /// Returns select and unselected registers.
    pub fn get_all_selected(self) -> Result<Vec<R>, Vec<R>> {
        match self {
            SplitManyResult::AllSelected(v) => Ok(v),
            SplitManyResult::Remaining(v, _) => Err(v),
        }
    }

    /// Returns select registers, throws out remaining.
    pub fn get_selected(self) -> Vec<R> {
        match self {
            SplitManyResult::AllSelected(v) => v,
            SplitManyResult::Remaining(v, _) => v,
        }
    }
}

/// A base-level circuit builder trait, requiring definitions of registers, base circuit objects,
/// and end-result quantum state.
pub trait CircuitBuilder {
    /// The register type used for the circuit.
    type Register: QubitRegister + Debug;
    /// The struct used to represent circuit objects.
    type CircuitObject;
    /// Return type for state calculations.
    type StateCalculation;

    /// Number of qubits in circuit.
    fn n(&self) -> usize;

    /// Construct a single qubit.
    fn qubit(&mut self) -> Self::Register {
        self.register(NonZeroUsize::new(1).unwrap())
    }

    /// Construct a register with multiple qubits. Fails if n=0.
    fn qudit(&mut self, n: usize) -> Option<Self::Register> {
        NonZeroUsize::new(n).map(|n| self.register(n))
    }

    /// Construct a register with multiple qubits.
    fn register(&mut self, n: NonZeroUsize) -> Self::Register;

    /// Construct a register with multiple qubits. Fails if n=0.
    fn try_register(&mut self, n: usize) -> Option<Self::Register> {
        self.qudit(n)
    }

    /// Merge two registers into a single register with first the r1 indices, then the r2 indices.
    fn merge_two_registers(&mut self, r1: Self::Register, r2: Self::Register) -> Self::Register;

    /// Merge multiple registers together into a single register, returns None if none given.
    fn merge_registers<It>(&mut self, rs: It) -> Option<Self::Register>
    where
        It: IntoIterator<Item = Self::Register>,
    {
        rs.into_iter().fold(None, |acc, r1| match acc {
            Some(r2) => Some(self.merge_two_registers(r2, r1)),
            None => Some(r1),
        })
    }

    /// Split a register into two, selecting the relative indices from the `indices` iterator.
    fn split_register_relative<It>(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitResult<Self::Register>
    where
        It: IntoIterator<Item = usize>;

    /// Split a register into two, selecting the indices from the `indices` iterator.
    fn split_register_absolute<It>(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitResult<Self::Register>
    where
        It: IntoIterator<Item = usize>,
    {
        let r_indices = r.indices().to_vec();
        let r_rel_indices = indices.into_iter().filter_map(move |abs_index| {
            // Ok to use n^2 since n must be small.
            r_indices.iter().cloned().find(|i| *i == abs_index)
        });
        self.split_register_relative(r, r_rel_indices)
    }

    /// Split the register into `r.n()` individual registers of 1 qubit each.
    fn split_all_register(&mut self, r: Self::Register) -> Vec<Self::Register> {
        split_helper(self, r, vec![])
    }

    /// Split off the first qubit from the register, returns the optional remaining registers
    /// and the first qubit.
    fn split_first_qubit(&mut self, r: Self::Register) -> (Option<Self::Register>, Self::Register) {
        match self.split_register_relative(r, [0]) {
            SplitResult::SELECTED(r) => (None, r),
            SplitResult::SPLIT(ra, rb) => (Some(ra), rb),
            SplitResult::UNSELECTED(_) => unreachable!(),
        }
    }

    /// Similar to [split_first_qubit] but the last qubit.
    fn split_last_qubit(&mut self, r: Self::Register) -> (Self::Register, Option<Self::Register>) {
        let n = r.n();
        match self.split_register_relative(r, [n - 1]) {
            SplitResult::SELECTED(r) => (r, None),
            SplitResult::SPLIT(ra, rb) => (rb, Some(ra)),
            SplitResult::UNSELECTED(_) => unreachable!(),
        }
    }

    /// Split into multiple qubits, each with relative indices given by the sub-iterators.
    ///
    /// # Example
    /// ```
    /// # use qip::prelude::*;
    ///
    /// # fn main() {
    /// let mut b = LocalBuilder::<f64>::default();
    /// let ra = b.qudit(5).expect("5 is non-negative");
    /// let rb = b.qudit(5).expect("5 is non-negative");
    /// assert_eq!(ra.indices(), &[0,1,2,3,4]);
    /// let split_res = b.split_relative_index_groups(rb, [[0,1], [2,3]]);
    /// if let SplitManyResult::Remaining(groups, remaining) = split_res {
    ///     assert_eq!(groups[0].indices(), &[5, 6]);
    ///     assert_eq!(groups[1].indices(), &[7, 8]);
    ///     assert_eq!(remaining.indices(), &[9])
    /// } else {
    ///     assert!(false);
    /// };
    ///
    /// # }
    /// ```
    fn split_relative_index_groups<
        It: IntoIterator<Item = Itt>,
        Itt: IntoIterator<Item = usize>,
    >(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitManyResult<Self::Register> {
        let mut rs = self
            .split_all_register(r)
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let selected_rs = indices
            .into_iter()
            .flat_map(|is| {
                let subrs = is.into_iter().map(|i| rs[i].take().unwrap());
                self.merge_registers(subrs)
            })
            .collect();
        let remaining_rs = self.merge_registers(rs.into_iter().flatten());
        match remaining_rs {
            None => SplitManyResult::AllSelected(selected_rs),
            Some(r) => SplitManyResult::Remaining(selected_rs, r),
        }
    }

    /// Apply a circuit object to the circuit directly.
    fn apply_circuit_object(
        &mut self,
        r: Self::Register,
        c: Self::CircuitObject,
    ) -> CircuitResult<Self::Register>;

    /// Calculate the quantum state at the end of the circuit, using |0> as input.
    fn calculate_state(&mut self) -> Self::StateCalculation {
        self.calculate_state_with_init(None)
    }

    /// Calculate the state at the end of the circuit using an initial state given by each register
    /// and the classical state in that register.
    fn calculate_state_with_init<'a, It>(&mut self, it: It) -> Self::StateCalculation
    where
        Self::Register: 'a,
        It: IntoIterator<Item = (&'a Self::Register, usize)>;
}

fn split_helper<CB>(cb: &mut CB, r: CB::Register, mut acc: Vec<CB::Register>) -> Vec<CB::Register>
where
    CB: CircuitBuilder + ?Sized,
{
    match cb.split_register_relative(r, Some(0)) {
        SplitResult::SELECTED(r) => {
            acc.push(r);
            acc
        }
        SplitResult::SPLIT(r0, r) => {
            acc.push(r0);
            split_helper(cb, r, acc)
        }
        SplitResult::UNSELECTED(_) => unreachable!(),
    }
}

/// Standard functions for building unitary circuits.
pub trait UnitaryBuilder<P: Precision>: CircuitBuilder {
    /// Apply an arbitrary matrix to the circuit given by a vector.
    fn apply_vec_matrix(
        &mut self,
        r: Self::Register,
        data: Vec<Complex<P>>,
    ) -> CircuitResult<Self::Register> {
        let n = r.n();
        self.apply_circuit_object(r, Self::vec_matrix_to_circuitobject(n, data))
    }

    /// Apply an arbitrary matrix to the circuit given by an array.
    fn apply_matrix<const N: usize>(
        &mut self,
        r: Self::Register,
        data: [Complex<P>; N],
    ) -> CircuitResult<Self::Register> {
        let n = r.n();
        self.apply_circuit_object(r, Self::matrix_to_circuitobject(n, data))
    }

    /// Single qubit matrices can be applied to each qubit in a register unambiguously.
    /// Matrix is organized as  |0><0|, |0><1|, |1><0|, |1><1|
    fn broadcast_single_qubit_matrix(
        &mut self,
        r: Self::Register,
        data: [Complex<P>; 4],
    ) -> Self::Register {
        let n = r.n();
        self.apply_circuit_object(r, Self::matrix_to_circuitobject(n, data))
            .unwrap()
    }

    /// Make a circuit object out of an arbitrary matrix
    /// Single Qubit matrix is organized as  |0><0|, |0><1|, |1><0|, |1><1|
    fn matrix_to_circuitobject<const N: usize>(
        n: usize,
        data: [Complex<P>; N],
    ) -> Self::CircuitObject {
        Self::vec_matrix_to_circuitobject(n, data.to_vec())
    }

    /// Make a circuit object out of an arbitrary matrix
    /// Single Qubit matrix is organized as  |0><0|, |0><1|, |1><0|, |1><1|
    fn vec_matrix_to_circuitobject(n: usize, data: Vec<Complex<P>>) -> Self::CircuitObject;
}

/// A Builder which can construct Clifford Circuit Elements.
pub trait CliffordTBuilder<P: Precision>: UnitaryBuilder<P> {
    /// Make a circuit object representing the X gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[0, 1, 1, 0\]
    fn make_x(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::zero(),
                Complex::one(),
                Complex::one(),
                Complex::zero(),
            ],
        )
    }

    /// Make a circuit object representing the Y gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[0, -i, i, 0\]
    fn make_y(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::zero(),
                -Complex::i(),
                Complex::i(),
                Complex::zero(),
            ],
        )
    }

    /// Make a circuit object representing the Z gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[1, 0, 0, -1\]
    fn make_z(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                -Complex::one(),
            ],
        )
    }

    /// Make a circuit object representing the H gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[1, 1, 1, -1\]/sqrt(2)
    fn make_h(&self) -> Self::CircuitObject {
        let l = Complex::one() * P::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
        Self::matrix_to_circuitobject(1, [l, l, l, -l])
    }

    /// Make a circuit object representing the S (phase) gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[1, 0, 0, -i\]
    fn make_s(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::i(),
            ],
        )
    }

    /// Make a circuit object representing the T gate on a single qubit.
    /// Equivalent to calling `matrix_to_circuitobject` with \[1, 0, 0, e^{i pi / 4} \]
    fn make_t(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::one(),
                Complex::zero(),
                Complex::zero(),
                Complex::from_polar(P::one(), P::from(std::f64::consts::FRAC_PI_4).unwrap()),
            ],
        )
    }

    /// Make a circuit object representing the CNOT gate on a pair of qubits
    /// Equivalent to calling `matrix_to_circuitobject` with \[ I, 0, 0, X \]
    /// where I is the identity matrix and X is the x-gate.
    fn make_cnot(&self) -> Self::CircuitObject {
        let l = Complex::one();
        let o = Complex::zero();
        Self::matrix_to_circuitobject(2, [l, o, o, o, o, l, o, o, o, o, o, l, o, o, l, o])
    }

    /// Create and apply an NOT (or X) gate circuit object.
    fn not(&mut self, r: Self::Register) -> Self::Register {
        self.x(r)
    }

    /// Create and apply an X (or NOT) gate circuit object.
    fn x(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_x()).unwrap()
    }

    /// Create and apply a Y gate circuit object.
    fn y(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_y()).unwrap()
    }

    /// Create and apply a Z gate circuit object.
    fn z(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_z()).unwrap()
    }

    /// Create and apply an H gate circuit object.
    fn h(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_h()).unwrap()
    }

    /// Create and apply a T gate circuit object.
    fn t(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_t()).unwrap()
    }

    /// Create and apply a T^\dagger gate circuit object.
    fn t_dagger(&mut self, r: Self::Register) -> Self::Register {
        let r = self.s_dagger(r);
        self.t(r)
    }

    /// Create and apply an S gate circuit object.
    fn s(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_s()).unwrap()
    }

    /// Create and apply an S^\dagger gate circuit object.
    fn s_dagger(&mut self, r: Self::Register) -> Self::Register {
        let r = self.z(r);
        self.s(r)
    }

    /// Create and apply a CNOT gate circuit object.
    fn cnot(
        &mut self,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        if cr.n() > 1 {
            Err(CircuitError::new(
                "Clifford CNOT can only have a single control qubit.",
            ))
        } else {
            let rs = self.split_all_register(r);
            let (cr, rs) = rs.into_iter().try_fold((cr, vec![]), |(cr, mut acc), r| {
                let r = self.merge_two_registers(cr, r);
                let circuit_object = self.make_cnot();
                let r = self.apply_circuit_object(r, circuit_object)?;
                let (cr, r) = match self.split_register_relative(r, Some(0)) {
                    SplitResult::SPLIT(cr, r) => (cr, r),
                    SplitResult::SELECTED(_) => unreachable!(),
                    SplitResult::UNSELECTED(_) => unreachable!(),
                };
                acc.push(r);
                Ok((cr, acc))
            })?;
            let r = self.merge_registers(rs).unwrap();
            Ok((cr, r))
        }
    }

    /// Apply the SWAP gate to a pair of registers of equal sizes.
    fn swap(
        &mut self,
        ra: Self::Register,
        rb: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        if ra.n() == rb.n() {
            let ras = self.split_all_register(ra);
            let rbs = self.split_all_register(rb);
            let (ras, rbs): (Vec<_>, Vec<_>) = ras
                .into_iter()
                .zip(rbs)
                .map(|(ra, rb)| {
                    assert_eq!(ra.n(), 1);
                    assert_eq!(rb.n(), 1);
                    self.cnot(ra, rb)
                        .and_then(|(ra, rb)| self.cnot(rb, ra))
                        .and_then(|(rb, ra)| self.cnot(ra, rb))
                        .unwrap()
                })
                .unzip();
            let ra = self.merge_registers(ras).unwrap();
            let rb = self.merge_registers(rbs).unwrap();
            Ok((ra, rb))
        } else {
            Err(CircuitError::new(
                "Swap must be between registers of the same size.",
            ))
        }
    }
}

/// A Builder which can construct temporary qudits.
pub trait TemporaryRegisterBuilder: CircuitBuilder {
    /// Make a temporary qubit, initialized to zero.
    fn make_zeroed_temp_qubit(&mut self) -> Self::Register;
    /// Make a register of multiple qubits, initialized to zero.
    fn make_zeroed_temp_register(&mut self, n: NonZeroUsize) -> Self::Register {
        let rs = (0..usize::from(n))
            .map(|_| self.make_zeroed_temp_qubit())
            .collect::<Vec<_>>();
        self.merge_registers(rs).unwrap()
    }
    /// Return a register which has been reset to zero.
    fn return_zeroed_temp_register(&mut self, r: Self::Register);
}

/// A builder which can construct more advanced gates using temporary qudits.
pub trait AdvancedCircuitBuilder<P: Precision>:
    CliffordTBuilder<P> + TemporaryRegisterBuilder
{
    /// Applies a NOT gate to `r` for the two qubit control state `cr = 11`.
    fn basic_toffoli(
        &mut self,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        if cr.n() == 2 {
            if let SplitResult::SPLIT(cra, crb) = self.split_register_relative(cr, [0]) {
                // Manually implement toffoli gate using CNOT
                let r = self.h(r);
                let (crb, r) = self.cnot(crb, r).unwrap();
                let r = self.t_dagger(r);
                let (cra, r) = self.cnot(cra, r).unwrap();
                let r = self.t(r);
                let (crb, r) = self.cnot(crb, r).unwrap();
                let r = self.t_dagger(r);
                let (cra, r) = self.cnot(cra, r).unwrap();
                let crb = self.t(crb);
                let r = self.t(r);
                let (cra, crb) = self.cnot(cra, crb).unwrap();
                let r = self.h(r);
                let cra = self.t(cra);
                let crb = self.t_dagger(crb);
                let (cra, crb) = self.cnot(cra, crb).unwrap();
                let cr = self.merge_two_registers(cra, crb);
                Ok((cr, r))
            } else {
                unreachable!()
            }
        } else {
            Err(CircuitError::new(
                "Basic Toffoli can only be applied to two control qubits.",
            ))
        }
    }

    /// Applies NOT to `r` if all qubits in `cr` are `1`.
    fn toffoli(
        &mut self,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        if cr.n() == 1 {
            self.cnot(cr, r)
        } else if cr.n() == 2 {
            self.basic_toffoli(cr, r)
        } else if let SplitResult::SPLIT(crhead, crtail) = self.split_register_relative(cr, [0, 1])
        {
            let tr = self.make_zeroed_temp_qubit();

            let (crhead, tr) = self.toffoli(crhead, tr).unwrap();

            let cr = self.merge_two_registers(crtail, tr);
            let (cr, r) = self.toffoli(cr, r).unwrap();

            let (crtail, tr) = self.split_last_qubit(cr);
            let tr = tr.unwrap();
            let (crhead, tr) = self.toffoli(crhead, tr).unwrap();

            self.return_zeroed_temp_register(tr);
            Ok((self.merge_two_registers(crhead, crtail), r))
        } else {
            unreachable!()
        }
    }
}

/// A Builder which can construct arbitrary rotations around axes.
pub trait RotationsBuilder<P: Precision>: CliffordTBuilder<P> {
    /// Rotate around z.
    fn rz(&mut self, r: Self::Register, theta: P) -> Self::Register;
    /// Rotate around x.
    fn rx(&mut self, r: Self::Register, theta: P) -> Self::Register {
        let r = self.h(r);
        let r = self.rz(r, theta);
        self.h(r)
    }
    /// Rotate around y.
    fn ry(&mut self, r: Self::Register, theta: P) -> Self::Register {
        let r = self.s_dagger(r);
        let r = self.h(r);
        let r = self.rz(r, -theta);
        let r = self.h(r);
        self.s(r)
    }
    /// Rotate around z.
    fn rz_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        Ok(self.rz(r, P::from(theta).unwrap()))
    }
    /// Rotate around z.
    fn rx_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        let r = self.h(r);
        let r = self.rz_ratio(r, theta)?;
        Ok(self.h(r))
    }
    /// Rotate around z.
    fn ry_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        let r = self.s(r);
        let r = self.h(r);
        let r = self.rz_ratio(r, -theta)?;
        let r = self.h(r);
        Ok(self.s_dagger(r))
    }
    /// Rotate around z by pi/m
    fn rz_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.rz_ratio(r, Rational64::new(1, m))
    }
    /// Rotate around x by pi/m
    fn rx_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.rx_ratio(r, Rational64::new(1, m))
    }
    /// Rotate around y by pi/m
    fn ry_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.ry_ratio(r, Rational64::new(1, m))
    }
}

/// A builder that can take destructive measurements.
pub trait MeasurementBuilder: CircuitBuilder {
    /// Handle which points to measurements.
    type MeasurementHandle;
    /// Take a measurement of `r`, return `r` and a handle to fetch the result later.
    fn measure(&mut self, r: Self::Register) -> (Self::Register, Self::MeasurementHandle);
}

/// A builder that can take nondestructive measurements.
pub trait StochasticMeasurementBuilder: CircuitBuilder {
    /// Handle which points to measurements.
    type StochasticMeasurementHandle;
    /// Take a measurement of `r`, return `r` and a handle to fetch the result later.
    fn measure_stochastic(
        &mut self,
        r: Self::Register,
    ) -> (Self::Register, Self::StochasticMeasurementHandle);
}

/// A builder which can export its circuit for use later, and can apply a circuit to itself.
pub trait Subcircuitable: CircuitBuilder {
    /// The export type for the circuit.
    type Subcircuit;

    /// Export the circuit as a subcircuit if able.
    fn make_subcircuit(&self) -> CircuitResult<Self::Subcircuit>;
    /// Append the subcircuit to the register `r`.
    fn apply_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        r: Self::Register,
    ) -> CircuitResult<Self::Register>;
}

/// Create a circuit for the circuit given by `r`.
pub fn make_circuit_matrix<CB, P, F>(cb: &mut CB, r: &CB::Register, f: F) -> Vec<Vec<Complex<P>>>
where
    CB: CircuitBuilder,
    P: Precision,
    F: Fn(CB::StateCalculation) -> Vec<Complex<P>>,
{
    (0..1 << r.n())
        .map(|indx| f(cb.calculate_state_with_init(Some((r, indx)))))
        .collect()
}
