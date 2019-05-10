use super::qubits::*;
use super::state_ops::*;
use crate::qip::state_ops::QubitOp::{ControlOp, MatrixOp};
use crate::qip::qubits::Parent::Owned;

type MeasuredResultReference = u32;

pub trait NonUnitaryBuilder {
    // Things like measure go here

    fn measure(&mut self, q: Qubit) -> MeasuredResultReference;

}

pub trait UnitaryBuilder {
    // Things like X, Y, Z/NOT, H, SWAP, ... go here

    fn make_builder_with_context(&self, q: Qubit) -> Result<ConditionalContextBuilder, String>;

    fn not(&mut self, q: Qubit) -> Qubit;
    fn not_op(&self, q: &Qubit) -> QubitOp;
}

pub struct OpBuilder {

}

impl OpBuilder {
    pub fn new() -> OpBuilder {
        OpBuilder{}
    }
}

impl NonUnitaryBuilder for OpBuilder {
    fn measure(&mut self, q: Qubit) -> u32 {
        unimplemented!()
    }
}

impl UnitaryBuilder for OpBuilder {
    fn make_builder_with_context(&self, q: Qubit) -> Result<ConditionalContextBuilder, String> {
        if q.indices.len() == 1 {
            let indx = q.indices[0];
            Result::Ok(ConditionalContextBuilder {
                parent_builder: self,
                conditioned_qubit: Some(q),
                conditioned_index: indx
            })
        } else {
            Result::Err(String::from("Conditional qubit must have n=1"))
        }
    }

    fn not(&mut self, q: Qubit) -> Qubit {
        let op = self.not_op(&q);
        let op_fn = make_op_fn(op);
        Qubit::merge_with_fn(vec![q], op_fn)
    }

    fn not_op(&self, q: &Qubit) -> QubitOp {
        MatrixOp(q.indices.len(), to_complex(&[0.0, 1.0, 1.0, 0.0]))
    }
}

pub struct ConditionalContextBuilder<'a> {
    parent_builder: &'a UnitaryBuilder,
    conditioned_qubit: Option<Qubit>,
    conditioned_index: u64,
}

impl<'a> ConditionalContextBuilder<'a> {
    pub fn release_qubit(self: Self) -> Qubit {
        match self.conditioned_qubit {
            Some(q) => q,
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }
}

impl<'a> UnitaryBuilder for ConditionalContextBuilder<'a> {
    fn make_builder_with_context(&self, q: Qubit) -> Result<ConditionalContextBuilder, String> {
        if q.indices.len() == 1 {
            let indx = q.indices[0];
            Result::Ok(ConditionalContextBuilder {
                parent_builder: self,
                conditioned_qubit: Some(q),
                conditioned_index: indx
            })
        } else {
            Result::Err(String::from("Conditional qubit must have n=1"))
        }
    }

    fn not(&mut self, q: Qubit) -> Qubit {
        let op = self.not_op(&q);
        let op_fn = make_op_fn(op);
        let cq = self.conditioned_qubit.take().unwrap();
        let q = Qubit::merge_with_fn(vec![cq, q], op_fn);
        let (cq, q) = Qubit::split(q, vec![self.conditioned_index]);
        self.conditioned_qubit = Some(cq);
        q
    }

    fn not_op(&self, q: &Qubit) -> QubitOp {
        ControlOp(self.conditioned_index,
                  Box::new(self.parent_builder.not_op(q)))
    }
}