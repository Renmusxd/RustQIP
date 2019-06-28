use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use num::complex::Complex;

use crate::pipeline::*;
use crate::state_ops::*;
use crate::types::Precision;

/// Possible relations to a parent qubit
pub enum Parent<P: Precision> {
    Owned(Vec<Qubit<P>>, Option<StateModifier<P>>),
    Shared(Rc<Qubit<P>>),
}

/// A qubit object, possible representing multiple physical qubit indices.
pub struct Qubit<P: Precision> {
    pub indices: Vec<u64>,
    pub parent: Option<Parent<P>>,
    pub id: u64,
}

impl<P: Precision> Qubit<P> {
    fn new(id: u64, indices: Vec<u64>) -> Result<Qubit<P>, &'static str> {
        if indices.len() == 0 {
            Err("Qubit must have nonzero number of indices.")
        } else {
            Ok(Qubit::<P> {
                indices,
                parent: None,
                id,
            })
        }
    }

    /// Merge qubits to for a new qubit object.
    pub fn merge_with_modifier(id: u64, qubits: Vec<Qubit<P>>, modifier: Option<StateModifier<P>>) -> Qubit<P> {
        let mut all_indices = Vec::new();

        for q in qubits.iter() {
            all_indices.extend(q.indices.iter());
        }
        all_indices.sort();

        Qubit::<P> {
            indices: all_indices,
            parent: Some(Parent::Owned(qubits, modifier)),
            id,
        }
    }

    /// Split the relative indices out of `q` into its own qubit, remaining live in second qubit.
    pub fn split(ida: u64, idb: u64, q: Qubit<P>, selected_indices: Vec<u64>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        for indx in &selected_indices {
            if indx < &0 {
                return Err("All indices for splitting must be above 0");
            } else if indx > &(q.indices.len() as u64) {
                return Err("All indices for splitting must be below q.n");
            }
        }
        let selected_indices: Vec<u64> = selected_indices.into_iter().map(|i| q.indices[i as usize]).collect();
        Self::split_absolute(ida, idb, q, selected_indices)
    }

    /// Split a qubit in two, with one having the indices in `selected_indices`
    pub fn split_absolute(ida: u64, idb: u64, q: Qubit<P>, selected_indices: Vec<u64>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        if selected_indices.len() == q.indices.len() {
            return Err("Cannot split out all indices into own qubit.");
        }
        for indx in &selected_indices {
            if !q.indices.contains(indx) {
                return Err("All indices must exist in qubit to be split.");
            }
        };

        let remaining = q.indices.clone()
            .into_iter()
            .filter(|x| !selected_indices.contains(x))
            .collect();
        let shared_parent = Rc::new(q);

        Ok((Qubit {
            indices: selected_indices,
            parent: Some(Parent::Shared(shared_parent.clone())),
            id: ida,
        }, Qubit {
            indices: remaining,
            parent: Some(Parent::Shared(shared_parent.clone())),
            id: idb,
        }))
    }

    /// Get number of qubits in this Qubit object
    pub fn n(&self) -> u64 {
        self.indices.len() as u64
    }
}

impl<P: Precision> std::cmp::Eq for Qubit<P> {}

impl<P: Precision> std::cmp::PartialEq for Qubit<P> {
    fn eq(&self, other: &Qubit<P>) -> bool {
        self.id == other.id
    }
}

impl<P: Precision> std::cmp::Ord for Qubit<P> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<P: Precision> std::cmp::PartialOrd for Qubit<P> {
    fn partial_cmp(&self, other: &Qubit<P>) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Precision> fmt::Debug for Qubit<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let int_strings = self.indices.iter()
            .map(|x| x.clone().to_string())
            .collect::<Vec<String>>();

        write!(f, "Qubit[{}][{}]", self.id.to_string(), int_strings.join(", "))
    }
}

pub struct QubitHandle {
    indices: Vec<u64>
}

impl QubitHandle {
    pub fn make_init_from_index<P: Precision>(&self, index: u64) -> Result<QubitInitialState<P>, &'static str> {
        if index < 1 << self.indices.len() as u64 {
            Ok((self.indices.clone(), InitialState::Index(index)))
        } else {
            Err("Index too large for QubitHandle")
        }
    }
    pub fn make_init_from_state<P: Precision>(&self, state: Vec<Complex<P>>) -> Result<QubitInitialState<P>, &'static str> {
        if state.len() == 1 << self.indices.len() {
            Ok((self.indices.clone(), InitialState::FullState(state)))
        } else {
            Err("State not correct size for QubitHandle (must be 2^n)")
        }
    }
}

/// A builder which supports non-unitary operations
pub trait NonUnitaryBuilder<P: Precision> {
    /// Add a measure op to the pipeline for `q` and return a reference which can
    /// later be used to access the measured value from the results of `pipeline::run`.
    fn measure(&mut self, q: Qubit<P>) -> (Qubit<P>, u64);
}

/// A builder which support unitary operations
pub trait UnitaryBuilder<P: Precision> {
    // Things like X, Y, Z, NOT, H, SWAP, ... go here

    /// Build a builder which uses `q` as context.
    fn with_context(&mut self, q: Qubit<P>) -> ConditionalContextBuilder<P>;

    /// Build a generic matrix op, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^(2n)).
    fn mat(&mut self, q: Qubit<P>, mat: &[Complex<P>]) -> Result<Qubit<P>, &'static str>;

    /// Build a matrix op from real numbers, apply to `q`, if `q` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `q` (mat.len() == 2^(2n)).
    fn real_mat(&mut self, q: Qubit<P>, mat: &[P]) -> Result<Qubit<P>, &'static str> {
        self.mat(q, from_reals(mat).as_slice())
    }

    /// Apply NOT to `q`, if `q` is multiple indices, apply to each
    fn not(&mut self, q: Qubit<P>) -> Qubit<P> {
        self.x(q)
    }

    /// Apply X to `q`, if `q` is multiple indices, apply to each
    fn x(&mut self, q: Qubit<P>) -> Qubit<P> {
        self.real_mat(q, &[P::zero(), P::one(), P::one(), P::zero()]).unwrap()
    }

    /// Apply Y to `q`, if `q` is multiple indices, apply to each
    fn y(&mut self, q: Qubit<P>) -> Qubit<P> {
        self.mat(q, from_tuples(&[(P::zero(), P::zero()), (P::zero(), P::zero() - P::one()), (P::zero(), P::zero()), (P::zero(), P::one())])
            .as_slice()).unwrap()
    }

    /// Apply Z to `q`, if `q` is multiple indices, apply to each
    fn z(&mut self, q: Qubit<P>) -> Qubit<P> {
        self.real_mat(q, &[P::one(), P::zero(), P::zero(), P::zero()-P::one()]).unwrap()
    }

    /// Apply H to `q`, if `q` is multiple indices, apply to each
    fn hadamard(&mut self, q: Qubit<P>) -> Qubit<P> {
        let inv_sqrt = 1.0f64 / 2.0_f64.sqrt();
        let inv_sqrt = P::from(inv_sqrt).unwrap();
        self.real_mat(q, &[inv_sqrt, inv_sqrt, inv_sqrt, P::zero()-inv_sqrt]).unwrap()
    }

    /// Apply SWAP to `qa` and `qb`
    fn swap(&mut self, qa: Qubit<P>, qb: Qubit<P>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        let op = self.make_swap_op(&qa, &qb)?;
        let qa_indices = qa.indices.clone();
        let q = self.merge_with_op(vec![qa, qb], Some(op));
        self.split_absolute(q, qa_indices)
    }

    /// Merge the qubits in `qs` into a single qubit.
    fn merge(&mut self, qs: Vec<Qubit<P>>) -> Qubit<P> {
        self.merge_with_op(qs, None)
    }

    /// Split the qubit `q` into two qubits, one with `selected_indices` and one with the remaining.
    fn split_absolute(&mut self, q: Qubit<P>, selected_indices: Vec<u64>) -> Result<(Qubit<P>, Qubit<P>), &'static str>;

    fn split_absolute_many(&mut self, q: Qubit<P>, index_groups: Vec<Vec<u64>>) -> Result<(Vec<Qubit<P>>, Qubit<P>), &'static str> {
        Ok(index_groups.into_iter().fold((vec![], q), |(mut qs, q), indices| {
            let (hq, tq) = self.split_absolute(q, indices).unwrap();
            qs.push(hq);
            (qs, tq)
        }))
    }

    /// Split `q` into a single qubit for each index.
    fn split_all(&mut self, q: Qubit<P>) -> Vec<Qubit<P>> {
        let mut indices: Vec<Vec<u64>> = q.indices.iter().cloned().map(|i| vec![i]).collect();
        indices.pop();
        // Cannot fail since all indices are from q.
        let (mut qs, q) = self.split_absolute_many(q, indices).unwrap();
        qs.push(q);
        qs
    }

    /// Build a generic matrix op.
    fn make_mat_op(&self, q: &Qubit<P>, data: Vec<Complex<P>>) -> QubitOp<P> {
        QubitOp::MatrixOp(q.indices.clone(), data)
    }

    /// Build a swap op. qa and qb must have the same number of indices.
    fn make_swap_op(&self, qa: &Qubit<P>, qb: &Qubit<P>) -> Result<QubitOp<P>, &'static str> {
        if qa.indices.len() == qb.indices.len() {
            Ok(QubitOp::SwapOp(qa.indices.clone(), qb.indices.clone()))
        } else {
            Err("Swap must be made from two qubits of equal size.")
        }
    }

    /// Merge qubits using a generic state processing function.
    fn merge_with_op(&mut self, qs: Vec<Qubit<P>>, operator: Option<QubitOp<P>>) -> Qubit<P>;
}

/// A basic builder for unitary and non-unitary ops.
pub struct OpBuilder<P: Precision> {
    qubit_index: u64,
    op_id: u64,
    phantom: PhantomData<P>
}

impl<P: Precision> OpBuilder<P> {
    /// Build a new OpBuilder
    pub fn new() -> OpBuilder<P> {
        OpBuilder::<P> {
            qubit_index: 0,
            op_id: 0,
            phantom: PhantomData
        }
    }

    /// Build a new qubit with `n` indices
    pub fn qubit(&mut self, n: u64) -> Result<Qubit<P>, &'static str> {
        if n == 0 {
            Err("Qubit n must be greater than 0.")
        } else {
            let base_index = self.qubit_index;
            self.qubit_index = self.qubit_index + n;

            Qubit::new(self.get_op_id(), (base_index..self.qubit_index).collect())
        }
    }

    /// Build a new qubit with `n` indices, return it plus a handle which can be
    /// used for feeding in an initial state.
    pub fn qubit_and_handle(&mut self, n: u64) -> Result<(Qubit<P>, QubitHandle), &'static str> {
        let q = self.qubit(n)?;
        let indices = q.indices.clone();
        Ok((q, QubitHandle{ indices }))
    }

    fn get_op_id(&mut self) -> u64 {
        let tmp = self.op_id;
        self.op_id += 1;
        tmp
    }
}

impl<P: Precision> NonUnitaryBuilder<P> for OpBuilder<P> {
    fn measure(&mut self, q: Qubit<P>) -> (Qubit<P>, u64) {
        let id = self.get_op_id();
        let modifier = StateModifier::new_measurement(String::from("measure"), id.clone(), q.indices.clone());
        let modifier = Some(modifier);
        let q = Qubit::merge_with_modifier(id.clone(), vec![q], modifier);
        (q, id)
    }
}

impl<P: Precision> UnitaryBuilder<P> for OpBuilder<P> {
    fn with_context(&mut self, q: Qubit<P>) -> ConditionalContextBuilder<P> {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_qubit: Some(q),
        }
    }

    fn mat(&mut self, q: Qubit<P>, mat: &[Complex<P>]) -> Result<Qubit<P>, &'static str> {
        // Special case for broadcasting ops
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs.into_iter().map(|q| self.mat(q, mat).unwrap()).collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let expected_mat_size = 1 << (2*q.indices.len());
            if expected_mat_size != mat.len() {
                Err("Matrix not of expected size")
            } else {
                let op = self.make_mat_op(&q, mat.to_vec());
                Ok(self.merge_with_op(vec![q], Some(op)))
            }
        }
    }

    fn split_absolute(&mut self, q: Qubit<P>, selected_indices: Vec<u64>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        Qubit::split_absolute(self.get_op_id(), self.get_op_id(), q, selected_indices)
    }

    fn merge_with_op(&mut self, qs: Vec<Qubit<P>>, op: Option<QubitOp<P>>) -> Qubit<P> {
        let modifier = op.map(|op|StateModifier::new_unitary(String::from("unitary"), op));
        Qubit::merge_with_modifier(self.get_op_id(), qs, modifier)
    }
}

/// An op builder which depends on the value of a given qubit (COPs)
pub struct ConditionalContextBuilder<'a, P: Precision> {
    parent_builder: &'a mut UnitaryBuilder<P>,
    conditioned_qubit: Option<Qubit<P>>,
}

impl<'a, P: Precision> ConditionalContextBuilder<'a, P> {
    /// Release the qubit used to build this builder
    pub fn release_qubit(self: Self) -> Qubit<P> {
        match self.conditioned_qubit {
            Some(q) => q,
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }

    fn get_conditional_qubit(&mut self) -> Qubit<P> {
        self.conditioned_qubit.take().unwrap()
    }

    fn set_conditional_qubit(&mut self, cq: Qubit<P>) {
        self.conditioned_qubit = Some(cq);
    }
}

impl<'a, P: Precision> UnitaryBuilder<P> for ConditionalContextBuilder<'a, P> {
    fn with_context(&mut self, q: Qubit<P>) -> ConditionalContextBuilder<P> {
        ConditionalContextBuilder::<P> {
            parent_builder: self,
            conditioned_qubit: Some(q),
        }
    }

    fn mat(&mut self, q: Qubit<P>, mat: &[Complex<P>]) -> Result<Qubit<P>, &'static str> {
        // Special case for applying mat to each qubit in collection.
        if q.indices.len() > 1 && mat.len() == (2 * 2) {
            let qs = self.split_all(q);
            let qs = qs.into_iter().map(|q| self.mat(q, mat).unwrap()).collect();
            Ok(self.merge_with_op(qs, None))
        } else {
            let expected_mat_size = 1 << (2*q.indices.len());
            if expected_mat_size != mat.len() {
                Err("Matrix not of expected size")
            } else {
                let op = self.make_mat_op(&q, mat.to_vec());
                let cq = self.get_conditional_qubit();
                let cq_indices = cq.indices.clone();
                let q = self.merge_with_op(vec![cq, q], Some(op));
                let (cq, q) = self.split_absolute(q, cq_indices).unwrap();

                self.set_conditional_qubit(cq);
                Ok(q)
            }
        }
    }

    fn swap(&mut self, qa: Qubit<P>, qb: Qubit<P>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        let op = self.make_swap_op(&qa, &qb)?;
        let cq = self.get_conditional_qubit();
        let cq_indices = cq.indices.clone();
        let qa_indices = qa.indices.clone();
        let q = self.merge_with_op(vec![cq, qa, qb], Some(op));
        let (cq, q) = self.split_absolute(q, cq_indices).unwrap();
        let (qa, qb) = self.split_absolute(q, qa_indices).unwrap();

        self.set_conditional_qubit(cq);
        Ok((qa, qb))
    }

    fn split_absolute(&mut self, q: Qubit<P>, selected_indices: Vec<u64>) -> Result<(Qubit<P>, Qubit<P>), &'static str> {
        self.parent_builder.split_absolute(q, selected_indices)
    }

    fn make_mat_op(&self, q: &Qubit<P>, data: Vec<Complex<P>>) -> QubitOp<P> {
        match &self.conditioned_qubit {
            Some(cq) => make_control_op(cq.indices.clone(), self.parent_builder.make_mat_op(q, data)),
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }

    fn make_swap_op(&self, qa: &Qubit<P>, qb: &Qubit<P>) -> Result<QubitOp<P>, &'static str> {
        match &self.conditioned_qubit {
            Some(cq) => {
                let op = self.parent_builder.make_swap_op(qa, qb)?;
                Ok(make_control_op(cq.indices.clone(), op))
            },
            None => panic!("Conditional context builder failed to populate qubit.")
        }
    }

    fn merge_with_op(&mut self, qs: Vec<Qubit<P>>, op: Option<QubitOp<P>>) -> Qubit<P> {
        self.parent_builder.merge_with_op(qs, op)
    }
}