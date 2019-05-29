extern crate num;
use super::qubits::*;
use super::state_ops::*;
use super::pipeline::*;
use super::pipeline::MeasuredResultReference;
use num::complex::Complex;

pub trait NonUnitaryBuilder {
    // Things like measure go here
    fn measure(&mut self, q: Qubit) -> MeasuredResultReference;
}

pub trait UnitaryBuilder {
    // Things like X, Y, Z, NOT, H, SWAP, ... go here

    fn with_context(&self, q: Qubit) -> ConditionalContextBuilder;

    fn mat(&mut self, q: Qubit, mat: &[Complex<f64>]) -> Qubit;

    fn real_mat(&mut self, q: Qubit, mat: &[f64]) -> Qubit {
        self.mat(q, from_reals(mat).as_slice())
    }

    fn not(&mut self, q: Qubit) -> Qubit {
        self.x(q)
    }

    fn x(&mut self, q: Qubit) -> Qubit {
        self.real_mat(q, &[0.0, 1.0, 1.0, 0.0])
    }

    fn y(&mut self, q: Qubit) -> Qubit {
        self.mat(q, from_tuples(&[(0.0,0.0), (0.0, -1.0), (0.0, 0.0), (0.0, 1.0)])
            .as_slice())
    }

    fn z(&mut self, q: Qubit) -> Qubit {
        self.real_mat(q, &[1.0, 0.0, 0.0, -1.0])
    }

    fn hadamard(&mut self, q: Qubit) -> Qubit {
        self.real_mat(q, &[1.0, 1.0, -1.0, 1.0])
    }

    fn swap(&mut self, qa: Qubit, qb: Qubit) -> (Qubit, Qubit) {
        unimplemented!()
    }

    fn make_mat_op(&self, q: &Qubit, data: Vec<Complex<f64>>) -> QubitOp {
        QubitOp::MatrixOp(q.indices.len(), data)
    }

    // Swap can be optimized so have that as an option.
    fn make_swap_op(&self, qa: &Qubit, qb: &Qubit) -> QubitOp {
        QubitOp::SwapOp(qa.indices.clone(), qb.indices.clone())
    }
}

pub struct OpBuilder {

}

impl OpBuilder {
    pub fn new() -> OpBuilder {
        OpBuilder{}
    }
}

impl NonUnitaryBuilder for OpBuilder {
    fn measure(&mut self, q: Qubit) -> MeasuredResultReference {
        unimplemented!()
    }
}

impl UnitaryBuilder for OpBuilder {
    fn with_context(&self, q: Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: Some(q)
        }
    }

    fn mat(&mut self, q: Qubit, mat: &[Complex<f64>]) -> Qubit {
        let op = self.make_mat_op(&q, mat.to_vec());
        let op_fn = make_op_fn(op);
        Qubit::merge_with_fn(vec![q], Some(op_fn))
    }
}

pub struct ConditionalContextBuilder<'a> {
    parent_builder: &'a UnitaryBuilder,
    conditioned_qubit: Option<Qubit>,
}

impl<'a> ConditionalContextBuilder<'a> {
    pub fn release_qubit(self: Self) -> Qubit {
        match self.conditioned_qubit {
            Some(q) => q,
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }

    fn get_conditional_qubit(&mut self) -> Qubit {
        self.conditioned_qubit.take().unwrap()
    }

    fn set_conditional_qubit(&mut self, cq: Qubit) {
        self.conditioned_qubit = Some(cq);
    }
}

impl<'a> UnitaryBuilder for ConditionalContextBuilder<'a> {

    fn with_context(&self, q: Qubit) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: Some(q)
        }
    }

    fn mat(&mut self, q: Qubit, mat: &[Complex<f64>]) -> Qubit {
        let op = self.make_mat_op(&q, mat.to_vec());
        let op_fn = make_op_fn(op);

        let cq = self.get_conditional_qubit();
        let cq_indices = cq.indices.clone();
        let q = Qubit::merge_with_fn(vec![cq, q], Some(op_fn));
        let (cq, q) = Qubit::split(q, cq_indices);

        self.set_conditional_qubit(cq);
        q
    }

    fn make_mat_op(&self, q: &Qubit, data: Vec<Complex<f64>>) -> QubitOp {
        match &self.conditioned_qubit {
            Some(cq) => QubitOp::ControlOp(cq.indices.clone(),
                                          Box::new(self.parent_builder.make_mat_op(q, data))),
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }

    fn make_swap_op(&self, qa: &Qubit, qb: &Qubit) -> QubitOp {
        unimplemented!()
    }
}