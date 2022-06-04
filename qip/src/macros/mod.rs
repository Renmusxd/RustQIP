use crate::builder_traits::{AdvancedCircuitBuilder, Subcircuitable};
use crate::conditioning::{Conditionable, ConditionableSubcircuit};
use crate::inverter::Invertable;
use crate::Precision;

pub mod program;
pub mod program_ops;

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
