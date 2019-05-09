use crate::qip;

pub trait NonUnitaryBuilder {
    // Things like measure go here
}

pub trait UnitaryBuilder {
    // Things like X, Y, Z/NOT, H, SWAP, ... go here

    fn make_builder_with_context(self: &Self, q: qip::Qubit) -> ConditionalContextBuilder;

}

pub struct OpBuilder {

}

struct ConditionalContextBuilder<'a> {
    parent_builder: &'a UnitaryBuilder,
    conditioned_qubit: qip::Qubit,
}

impl OpBuilder {
    pub fn new() -> OpBuilder {
        OpBuilder{}
    }
}

impl NonUnitaryBuilder for OpBuilder {

}

impl UnitaryBuilder for OpBuilder {
    fn make_builder_with_context(self: &Self, q: qip::Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: q
        }
    }
}


impl<'a> ConditionalContextBuilder<'a> {
    fn release_qubit(self) -> qip::Qubit {
        self.conditioned_qubit
    }
}

impl<'a> UnitaryBuilder for ConditionalContextBuilder<'a> {
    fn make_builder_with_context(self: &Self, q: qip::Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: q
        }
    }
}