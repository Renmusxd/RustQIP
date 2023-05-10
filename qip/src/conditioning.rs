use crate::builder_traits::{QubitRegister, SplitResult, Subcircuitable};
use crate::errors::{CircuitError, CircuitResult};
use crate::inverter::Invertable;
use crate::inverter::RecursiveCircuitBuilder;
use crate::prelude::*;
use crate::types::Precision;
use num_complex::Complex;
use num_rational::Rational64;
use std::num::NonZeroUsize;

/// A CircuitBuilder is conditionable if it can condition all unitaries with a given register.
pub trait Conditionable: CircuitBuilder {
    /// Attempt to condition a circuit object `co` applied to `r` with the register `cr`.
    fn try_apply_with_condition(
        &mut self,
        cr: Self::Register,
        r: Self::Register,
        co: Self::CircuitObject,
    ) -> Result<(Self::Register, Self::Register), CircuitError>;

    /// Construct a new circuitbuilder which conditions all unitaries with `cr`.
    fn condition_with(&mut self, cr: Self::Register) -> Conditioned<Self> {
        Conditioned::new(self, cr)
    }
}

/// A CircuitBuilder which conditions all unitaries with a given register.
#[derive(Debug)]
pub struct Conditioned<'a, CB: Conditionable + ?Sized> {
    parent: &'a mut CB,
    cr: Option<CB::Register>,
}

impl<'a, CB: Conditionable + ?Sized> Conditioned<'a, CB> {
    fn new(cb: &'a mut CB, cr: CB::Register) -> Self {
        Self {
            parent: cb,
            cr: Some(cr),
        }
    }

    /// Dissolve the Conditioned circuit builder and retrieve the conditioning register.
    pub fn dissolve(self) -> CB::Register {
        self.cr.unwrap()
    }
}

impl<'a, CB: Conditionable + ?Sized> CircuitBuilder for Conditioned<'a, CB> {
    type Register = CB::Register;
    type CircuitObject = CB::CircuitObject;
    type StateCalculation = CB::StateCalculation;

    fn n(&self) -> usize {
        self.parent.n()
    }

    fn register(&mut self, n: NonZeroUsize) -> Self::Register {
        self.parent.register(n)
    }

    fn merge_two_registers(&mut self, r1: Self::Register, r2: Self::Register) -> Self::Register {
        self.parent.merge_two_registers(r1, r2)
    }

    fn split_register_relative<It>(
        &mut self,
        r: Self::Register,
        indices: It,
    ) -> SplitResult<Self::Register>
    where
        It: IntoIterator<Item = usize>,
    {
        self.parent.split_register_relative(r, indices)
    }

    fn apply_circuit_object(
        &mut self,
        r: Self::Register,
        c: Self::CircuitObject,
    ) -> CircuitResult<Self::Register> {
        let cr = self.cr.take().unwrap();
        let (cr, r) = self.parent.try_apply_with_condition(cr, r, c)?;
        self.cr = Some(cr);
        Ok(r)
    }

    fn calculate_state_with_init<'b, It>(&mut self, it: It) -> Self::StateCalculation
    where
        Self::Register: 'b,
        It: IntoIterator<Item = (&'b Self::Register, usize)>,
    {
        self.parent.calculate_state_with_init(it)
    }
}

impl<'a, P: Precision, CB: Conditionable + UnitaryBuilder<P> + ?Sized> UnitaryBuilder<P>
    for Conditioned<'a, CB>
{
    fn vec_matrix_to_circuitobject(n: usize, data: Vec<Complex<P>>) -> Self::CircuitObject {
        CB::vec_matrix_to_circuitobject(n, data)
    }
}

impl<'a, P: Precision, CB: Conditionable + CliffordTBuilder<P> + ?Sized> CliffordTBuilder<P>
    for Conditioned<'a, CB>
{
    fn make_x(&self) -> Self::CircuitObject {
        self.parent.make_x()
    }
    fn make_y(&self) -> Self::CircuitObject {
        self.parent.make_y()
    }
    fn make_z(&self) -> Self::CircuitObject {
        self.parent.make_z()
    }
    fn make_h(&self) -> Self::CircuitObject {
        self.parent.make_h()
    }
    fn make_s(&self) -> Self::CircuitObject {
        self.parent.make_s()
    }
    fn make_t(&self) -> Self::CircuitObject {
        self.parent.make_t()
    }
    fn make_cnot(&self) -> Self::CircuitObject {
        self.parent.make_cnot()
    }
}

impl<'a, P: Precision, CB: Conditionable + RotationsBuilder<P> + ?Sized> RotationsBuilder<P>
    for Conditioned<'a, CB>
{
    fn rz(&mut self, r: Self::Register, theta: P) -> Self::Register {
        self.parent.rz(r, theta)
    }

    fn rx(&mut self, r: Self::Register, theta: P) -> Self::Register {
        self.parent.rx(r, theta)
    }

    fn ry(&mut self, r: Self::Register, theta: P) -> Self::Register {
        self.parent.ry(r, theta)
    }

    fn rz_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        self.parent.rz_ratio(r, theta)
    }

    fn rx_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        self.parent.rx_ratio(r, theta)
    }

    fn ry_ratio(&mut self, r: Self::Register, theta: Rational64) -> CircuitResult<Self::Register> {
        self.parent.ry_ratio(r, theta)
    }

    fn rz_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.parent.rz_pi_by(r, m)
    }

    fn rx_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.parent.rx_pi_by(r, m)
    }

    fn ry_pi_by(&mut self, r: Self::Register, m: i64) -> CircuitResult<Self::Register> {
        self.parent.ry_pi_by(r, m)
    }
}

impl<'a, CB: Conditionable + TemporaryRegisterBuilder + ?Sized> TemporaryRegisterBuilder
    for Conditioned<'a, CB>
{
    fn make_zeroed_temp_qubit(&mut self) -> Self::Register {
        self.parent.make_zeroed_temp_qubit()
    }

    fn return_zeroed_temp_register(&mut self, r: Self::Register) {
        self.parent.return_zeroed_temp_register(r)
    }
}

impl<'a, P: Precision, CB: Conditionable + AdvancedCircuitBuilder<P> + ?Sized>
    AdvancedCircuitBuilder<P> for Conditioned<'a, CB>
{
}

impl<'a, CB: Conditionable> Conditionable for Conditioned<'a, CB> {
    fn try_apply_with_condition(
        &mut self,
        cr: CB::Register,
        r: CB::Register,
        co: CB::CircuitObject,
    ) -> Result<(CB::Register, CB::Register), CircuitError> {
        let ncr = cr.n();
        let ccr = self.cr.take().unwrap();
        let cr = self.merge_two_registers(cr, ccr);
        let (cr, r) = self.parent.try_apply_with_condition(cr, r, co)?;
        let split = self.split_register_relative(cr, 0..ncr);
        let (cr, ccr) = match split {
            SplitResult::SPLIT(cr, ccr) => (cr, ccr),
            SplitResult::SELECTED(_) => unreachable!(),
            SplitResult::UNSELECTED(_) => unreachable!(),
        };
        self.cr = Some(ccr);
        Ok((cr, r))
    }
}

/// A ConditionableSubcircuit may apply an entire subcircuit under the condition of `cr`.
pub trait ConditionableSubcircuit: Subcircuitable {
    /// Apply `sc` to register `r` using condition `cr`.
    fn apply_conditioned_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError>;
}

impl<'a, CB: ConditionableSubcircuit + Conditionable> Subcircuitable for Conditioned<'a, CB> {
    type Subcircuit = CB::Subcircuit;

    fn make_subcircuit(&self) -> CircuitResult<Self::Subcircuit> {
        self.parent.make_subcircuit()
    }

    fn apply_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        r: Self::Register,
    ) -> CircuitResult<Self::Register> {
        let cr = self.cr.take().unwrap();
        let (cr, r) = self.parent.apply_conditioned_subcircuit(sc, cr, r)?;
        self.cr = Some(cr);
        Ok(r)
    }
}

impl<'a, CB: Invertable + ConditionableSubcircuit + Conditionable> Invertable
    for Conditioned<'a, CB>
{
    type SimilarBuilder = CB::SimilarBuilder;

    fn new_similar(&self) -> Self::SimilarBuilder {
        self.parent.new_similar()
    }

    fn invert_subcircuit(sc: Self::Subcircuit) -> CircuitResult<Self::Subcircuit> {
        CB::invert_subcircuit(sc)
    }
}

impl<'a, CB: Invertable + ConditionableSubcircuit + Conditionable> ConditionableSubcircuit
    for Conditioned<'a, CB>
{
    fn apply_conditioned_subcircuit(
        &mut self,
        sc: Self::Subcircuit,
        cr: Self::Register,
        r: Self::Register,
    ) -> Result<(Self::Register, Self::Register), CircuitError> {
        let ncr = cr.n();
        let ccr = self.cr.take().unwrap();
        let cr = self.merge_two_registers(cr, ccr);
        let (cr, r) = self.parent.apply_conditioned_subcircuit(sc, cr, r)?;
        let split = self.split_register_relative(cr, 0..ncr);
        let (cr, ccr) = match split {
            SplitResult::SPLIT(cr, ccr) => (cr, ccr),
            SplitResult::SELECTED(_) => unreachable!(),
            SplitResult::UNSELECTED(_) => unreachable!(),
        };
        self.cr = Some(ccr);
        Ok((cr, r))
    }
}

impl<'a, P: Precision, CB: RecursiveCircuitBuilder<P>> RecursiveCircuitBuilder<P>
    for Conditioned<'a, CB>
where
    <CB as Invertable>::SimilarBuilder: RecursiveCircuitBuilder<P>,
{
    type RecursiveSimilarBuilder = Self::SimilarBuilder;
}
