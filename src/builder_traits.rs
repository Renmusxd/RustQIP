use crate::errors::{CircuitError, CircuitResult};
use crate::types::Precision;
use crate::Complex;
use num_rational::Rational64;
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::num::NonZeroUsize;

pub trait QubitRegister {
    fn n(&self) -> usize;
    fn n_nonzero(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.n()).unwrap()
    }
    fn indices(&self) -> &[usize];
}

#[derive(Debug)]
pub enum SplitResult<R: QubitRegister + Debug> {
    SELECTED(R),
    UNSELECTED(R),
    SPLIT(R, R),
}

#[derive(Debug)]
pub enum SplitManyResult<R: QubitRegister + Debug> {
    AllSelected(Vec<R>),
    Remaining(Vec<R>, R),
}

pub trait CircuitBuilder {
    type Register: QubitRegister + Debug;
    type CircuitObject;
    type StateCalculation;

    fn n(&self) -> usize;

    fn qubit(&mut self) -> Self::Register {
        self.register(NonZeroUsize::new(1).unwrap())
    }

    fn register(&mut self, n: NonZeroUsize) -> Self::Register;

    fn try_register(&mut self, n: usize) -> Option<Self::Register> {
        NonZeroUsize::new(n).map(|n| self.register(n))
    }

    fn merge_two_registers(&mut self, r1: Self::Register, r2: Self::Register) -> Self::Register;

    fn merge_registers<It>(&mut self, rs: It) -> Option<Self::Register>
    where
        It: IntoIterator<Item = Self::Register>,
    {
        rs.into_iter().fold(None, |acc, r1| match acc {
            Some(r2) => Some(self.merge_two_registers(r2, r1)),
            None => Some(r1),
        })
    }
    fn split_register_relative<It>(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitResult<Self::Register>
    where
        It: IntoIterator<Item = usize>;

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

    fn split_all_register(&mut self, r: Self::Register) -> Vec<Self::Register> {
        split_helper(self, r, vec![])
    }

    fn split_first_qubit(&mut self, r: Self::Register) -> (Option<Self::Register>, Self::Register) {
        match self.split_register_relative(r, [0]) {
            SplitResult::SELECTED(r) => (None, r),
            SplitResult::SPLIT(ra, rb) => (Some(ra), rb),
            SplitResult::UNSELECTED(_) => unreachable!(),
        }
    }

    fn split_last_qubit(&mut self, r: Self::Register) -> (Self::Register, Option<Self::Register>) {
        let n = r.n();
        match self.split_register_relative(r, [n - 1]) {
            SplitResult::SELECTED(r) => (r, None),
            SplitResult::SPLIT(ra, rb) => (rb, Some(ra)),
            SplitResult::UNSELECTED(_) => unreachable!(),
        }
    }

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
            .map(|is| {
                let subrs = is.into_iter().map(|i| rs[i].take().unwrap());
                self.merge_registers(subrs)
            })
            .flatten()
            .collect();
        let remaining_rs = self.merge_registers(rs.into_iter().flatten());
        match remaining_rs {
            None => SplitManyResult::AllSelected(selected_rs),
            Some(r) => SplitManyResult::Remaining(selected_rs, r),
        }
    }

    fn apply_circuit_object(
        &mut self,
        r: Self::Register,
        c: Self::CircuitObject,
    ) -> CircuitResult<Self::Register>;

    fn calculate_state(&mut self) -> Self::StateCalculation {
        self.calculate_state_with_init(None)
    }

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

pub trait UnitaryBuilder<P: Precision>: CircuitBuilder {
    fn apply_vec_matrix(
        &mut self,
        r: Self::Register,
        data: Vec<Complex<P>>,
    ) -> CircuitResult<Self::Register> {
        let n = r.n();
        self.apply_circuit_object(r, Self::vec_matrix_to_circuitobject(n, data))
    }

    fn apply_matrix<const N: usize>(
        &mut self,
        r: Self::Register,
        data: [Complex<P>; N],
    ) -> CircuitResult<Self::Register> {
        let n = r.n();
        self.apply_circuit_object(r, Self::matrix_to_circuitobject(n, data))
    }

    fn broadcast_single_qubit_matrix(
        &mut self,
        r: Self::Register,
        data: [Complex<P>; 4],
    ) -> Self::Register {
        let n = r.n();
        self.apply_circuit_object(r, Self::matrix_to_circuitobject(n, data))
            .unwrap()
    }

    fn matrix_to_circuitobject<const N: usize>(
        n: usize,
        data: [Complex<P>; N],
    ) -> Self::CircuitObject {
        Self::vec_matrix_to_circuitobject(n, data.to_vec())
    }

    fn vec_matrix_to_circuitobject(n: usize, data: Vec<Complex<P>>) -> Self::CircuitObject;
}

pub trait CliffordTBuilder<P: Precision>: UnitaryBuilder<P> {
    fn make_x(&self) -> Self::CircuitObject {
        Self::matrix_to_circuitobject(
            1,
            [
                Complex::zero(),
                Complex::one(),
                Complex::one(),
                -Complex::zero(),
            ],
        )
    }
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
    fn make_h(&self) -> Self::CircuitObject {
        let l = Complex::one() * P::from(std::f64::consts::FRAC_1_SQRT_2).unwrap();
        Self::matrix_to_circuitobject(1, [l, l, l, -l])
    }
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
    fn make_cnot(&self) -> Self::CircuitObject {
        let l = Complex::one();
        let o = Complex::zero();
        Self::matrix_to_circuitobject(2, [l, o, o, o, o, l, o, o, o, o, o, l, o, o, l, o])
    }

    fn not(&mut self, r: Self::Register) -> Self::Register {
        self.x(r)
    }
    fn x(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_x()).unwrap()
    }
    fn y(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_y()).unwrap()
    }
    fn z(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_z()).unwrap()
    }
    fn h(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_h()).unwrap()
    }
    fn t(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_t()).unwrap()
    }
    fn t_dagger(&mut self, r: Self::Register) -> Self::Register {
        let r = self.s_dagger(r);
        self.t(r)
    }
    fn s(&mut self, r: Self::Register) -> Self::Register {
        self.apply_circuit_object(r, self.make_s()).unwrap()
    }
    fn s_dagger(&mut self, r: Self::Register) -> Self::Register {
        let r = self.z(r);
        self.s(r)
    }
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
                .zip(rbs.into_iter())
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

pub trait TemporaryRegisterBuilder: CircuitBuilder {
    fn make_zeroed_temp_qubit(&mut self) -> Self::Register;
    fn make_zeroed_temp_register(&mut self, n: NonZeroUsize) -> Self::Register {
        let rs = (0..usize::from(n))
            .map(|_| self.make_zeroed_temp_qubit())
            .collect::<Vec<_>>();
        self.merge_registers(rs).unwrap()
    }
    fn return_zeroed_temp_register(&mut self, r: Self::Register);
}

pub trait AdvancedCircuitBuilder<P: Precision>:
    CliffordTBuilder<P> + TemporaryRegisterBuilder
{
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

pub trait RotationsBuilder<P: Precision>: CliffordTBuilder<P> {
    fn rz(&mut self, r: Self::Register, theta: P) -> Self::Register;
    fn rx(&mut self, r: Self::Register, theta: P) -> Self::Register {
        let r = self.h(r);
        let r = self.rz(r, theta);
        self.h(r)
    }
    fn ry(&mut self, r: Self::Register, theta: P) -> Self::Register {
        let r = self.s_dagger(r);
        let r = self.h(r);
        let r = self.rz(r, -theta);
        let r = self.h(r);
        self.s(r)
    }
    fn rz_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        Ok(self.rz(r, P::from(theta).unwrap()))
    }
    fn rx_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        let r = self.h(r);
        let r = self.rz_ratio(r, theta)?;
        Ok(self.h(r))
    }
    fn ry_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        let r = self.s(r);
        let r = self.h(r);
        let r = self.rz_ratio(r, -theta)?;
        let r = self.h(r);
        Ok(self.s_dagger(r))
    }
    fn rz_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.rz_ratio(r, Rational64::new(1, m))
    }
    fn rx_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.rx_ratio(r, Rational64::new(1, m))
    }
    fn ry_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.ry_ratio(r, Rational64::new(1, m))
    }
}

pub trait MeasurementBuilder: CircuitBuilder {
    type MeasurementHandle;
    fn measure(&mut self, r: Self::Register) -> (Self::Register, Self::MeasurementHandle);
}

pub trait StochasticMeasurementBuilder: CircuitBuilder {
    type StochasticMeasurementHandle;
    fn measure_stochastic(
        &mut self,
        r: Self::Register,
    ) -> (Self::Register, Self::StochasticMeasurementHandle);
}

pub trait Subcircuitable: CircuitBuilder {
    type Subcircuit;

    fn make_subcircuit(&self) -> CircuitResult<Self::Subcircuit>;
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
