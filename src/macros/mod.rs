use crate::builder_traits::{AdvancedCircuitBuilder, Subcircuitable};
use crate::conditioning::{Conditionable, ConditionableSubcircuit};
use crate::macros::inverter::Invertable;
use crate::Precision;

pub mod inverter;
pub mod program;
pub mod program_ops;
pub mod wrap_function;

pub trait RecursiveCircuitBuilder<P: Precision>:
    Invertable<SimilarBuilder = Self::RecursiveSimilarBuilder>
    + Conditionable
    + AdvancedCircuitBuilder<P>
    + Subcircuitable
    + Conditionable
    + ConditionableSubcircuit
{
    type RecursiveSimilarBuilder: RecursiveCircuitBuilder<P>;
}
