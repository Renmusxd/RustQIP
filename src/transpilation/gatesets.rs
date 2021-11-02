use crate::pipeline::{StateModifier, StateModifierType};
use crate::state_ops::UnitaryOp;
use crate::transpilation::traits::GateSet;
use crate::unitary_decomposition::decomposition::decompose_unitary;
use crate::{Register, UnitaryBuilder};
use num::{Complex, Zero};

struct ContinuousGateset<U: UnitaryBuilder> {
    b: U,
    registers: Vec<Register>,
}

#[derive(Copy, Clone)]
enum ContinuousGate {
    Rx(f64),
    Rz(f64),
    CNOT,
    I,
}

impl PartialEq for ContinuousGate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Rx(a), Self::Rx(b)) => (a - b).abs() < f64::EPSILON,
            (Self::Rz(a), Self::Rz(b)) => (a - b).abs() < f64::EPSILON,
            (Self::CNOT, Self::CNOT) => true,
            (_, _) => false,
        }
    }
}
impl Eq for ContinuousGate {}

impl<U: UnitaryBuilder> ContinuousGateset<U> {
    fn abc_decomposition(data: &[Complex<f64>]) -> [ContinuousGate; 3] {
        // Decompose into Rz(theta) Rx(phi) Rz(psi)
        // data[0,1] = i e^{i(theta-psi)} sin(phi)

        if data[0].norm() < f64::EPSILON {
            assert!(data[3].norm() < f64::EPSILON);
            // If all off-diagonal

            let phi = std::f64::consts::FRAC_PI_2;
            // Now i e^i{theta - psi} = data[2]
            // and i e^i{psi - theta} = data[3]
            // Underspecified - lets choose minimal gates.
            // psi = 0 --> e^{i theta} = - i data[2]
            let x = -data[2] * Complex::i();
            let theta = x.re.atan2(x.im);

            [
                ContinuousGate::Rz(theta),
                ContinuousGate::Rx(phi),
                ContinuousGate::I,
            ]
        } else if data[1].norm() < f64::EPSILON {
            assert!(data[2].norm() < f64::EPSILON);
            // All diagonal

            // Now e^i{theta + psi} = data[0]
            // Underspecified - lets choose minimal gates.
            // psi = 0 --> e^{i theta} = data[0]
            let x = data[0];
            let theta = x.re.atan2(x.im);
            [
                ContinuousGate::Rz(theta),
                ContinuousGate::I,
                ContinuousGate::I,
            ]
        } else {
            // General case
            let phi = data[1].norm().atan2(data[0].norm());
            // Now -i data[0]*data[1] = 1/2 sin(2 phi) exp(2 theta)
            // Solve for e^theta
            let exp_2theta: Complex<f64> =
                2. * (-Complex::i()) * data[0] * data[1] / (2. * phi).sin();
            let theta = exp_2theta.re.atan2(exp_2theta.im) / 2.;

            // Now -i data[0]*data[2] = 1/2 sin(2 phi) exp(theta)
            let exp_2psi: Complex<f64> =
                2. * (-Complex::i()) * data[0] * data[2] / (2. * phi).sin();
            let psi = exp_2psi.re.atan2(exp_2psi.im) / 2.;
            [
                ContinuousGate::Rz(theta),
                ContinuousGate::Rx(phi),
                ContinuousGate::Rz(psi),
            ]
        }
    }

    fn decompose_matrix(n: u64, data: &[Complex<f64>]) -> Vec<ContinuousGate> {
        let tn = 2_usize.pow(n as u32);
        // Special case the 1-qubit ops to use euler angles.
        if n == 1 {
            Self::abc_decomposition(data).to_vec()
        } else {
            const DEFAULT_DROP_MAG: f64 = 0.01;

            let mut sparse_data = vec![];
            for row in 0..tn {
                let mut row_data = vec![];
                for col in 0..tn {
                    let index = col * tn + row;
                    if data[index].norm() > f64::EPSILON {
                        row_data.push((col as u64, data[index]));
                    }
                }
                sparse_data.push(row_data);
            }
            Self::decompose_sparse(n, sparse_data)
        }
    }

    fn decompose_sparse(n: u64, data: Vec<Vec<(u64, Complex<f64>)>>) -> Vec<ContinuousGate> {
        if n == 1 {
            let mut dense_data = [Complex::zero(); 4];
            let tn = 2_usize.pow(n as u32);
            for (row, dat) in data.into_iter().enumerate() {
                for (col, val) in dat.into_iter() {
                    let index = col as usize * tn + row;
                    dense_data[index] = val;
                }
            }
            Self::decompose_matrix(n, &dense_data)
        } else {
            const DEFAULT_DROP_MAG: f64 = 1e-6;
            let res = decompose_unitary(n, data, DEFAULT_DROP_MAG);
            unimplemented!()
        }
    }

    fn compile_unitary(&self, op: &UnitaryOp) -> Vec<ContinuousGate> {
        match op {
            UnitaryOp::Matrix(indices, data) => {
                let decomp = Self::decompose_matrix(indices.len() as u64, data);
            }
            UnitaryOp::SparseMatrix(indices, data) => {}
            UnitaryOp::Swap(a_indices, b_indices) => {}
            UnitaryOp::Control(indices, _, op) => {}
            UnitaryOp::Function(a_indices, b_indices, f) => {}
        }
        todo!()
    }

    fn feed_unitary(&mut self, op: &UnitaryOp) {
        match op {
            UnitaryOp::Matrix(indices, data) => {}
            UnitaryOp::SparseMatrix(indices, data) => {}
            UnitaryOp::Swap(a_indices, b_indices) => {}
            UnitaryOp::Control(indices, op_indices, op) => {}
            UnitaryOp::Function(a_indices, b_indices, f) => {}
        }
        todo!()
    }
}

impl<U: UnitaryBuilder> GateSet<U> for ContinuousGateset<U> {
    fn new(b: U) -> Self {
        Self {
            b,
            registers: Vec::default(),
        }
    }

    fn feed(&mut self, u: &StateModifier) {
        match &u.modifier {
            StateModifierType::UnitaryOp(op) => self.feed_unitary(op),
            StateModifierType::MeasureState(id, indices, angle) => {}
            StateModifierType::StochasticMeasureState(id, indices, angle) => {}
            StateModifierType::SideChannelModifiers(handle, f) => {
                // We can wrap the function in a transpiler so that whatever it outputs is rewritten
                // in the apropriate gateset.
                todo!()
            }
            StateModifierType::Debug(indices, f) => {}
        }
    }

    fn dissolve(mut self) -> (U, Register) {
        let mut b = self.b;
        let r = b.merge(self.registers).unwrap();
        (b, r)
    }
}
