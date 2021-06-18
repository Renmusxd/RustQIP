#![feature(test)]

use qip::feynman_state::FeynmanState;
use qip::state_ops::UnitaryOp::*;
use qip::state_ops::*;
use qip::QuantumState;

#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;
    use test::Bencher;

    #[bench]
    fn bench_identity(b: &mut Bencher) {
        let n = 3;
        let indices = (0..n).collect::<Vec<_>>();
        let mut state = FeynmanState::<f64>::new(n);

        let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
        let ops: Vec<UnitaryOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
        ops.iter().for_each(|op| state.apply_op(op));

        b.iter(|| state.stochastic_measure(&indices, 0.0));
    }

    #[bench]
    fn bench_hadamard(b: &mut Bencher) {
        let n = 3;
        let indices = (0..n).collect::<Vec<_>>();
        let mut state = FeynmanState::<f64>::new(n);

        let mult = (1.0 / 2.0f64).sqrt();
        let mat = from_reals(&vec![mult, mult, mult, -mult]);
        let ops: Vec<UnitaryOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
        ops.iter().for_each(|op| state.apply_op(op));

        b.iter(|| state.stochastic_measure(&indices, 0.0));
    }

    #[bench]
    fn bench_cidentity(b: &mut Bencher) {
        let n = 3;
        let indices = (0..n).collect::<Vec<_>>();
        let mut state = FeynmanState::<f64>::new(n);

        let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
        let op = make_control_op((0..n - 1).collect(), Matrix(vec![n - 1], mat)).unwrap();
        state.apply_op(&op);

        b.iter(|| state.stochastic_measure(&indices, 0.0));
    }

    #[bench]
    fn bench_cidentity_giant(b: &mut Bencher) {
        let n = 16;
        let indices = (0..n).collect::<Vec<_>>();
        let mut state = FeynmanState::<f64>::new(n);

        let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
        let c_indices = (0..n - 1).collect();
        let op = make_matrix_op(vec![n - 1], mat).unwrap();
        let op = make_control_op(c_indices, op).unwrap();
        state.apply_op(&op);

        b.iter(|| state.stochastic_measure(&indices, 0.0));
    }

    #[bench]
    fn bench_hadamard_larger(b: &mut Bencher) {
        let n = 8;
        let indices = (0..n).collect::<Vec<_>>();
        let mut state = FeynmanState::<f64>::new(n);

        let mult = (1.0 / 2.0f64).sqrt();
        let mat = from_reals(&vec![mult, mult, mult, -mult]);

        let ops: Vec<UnitaryOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
        ops.iter().for_each(|op| state.apply_op(op));

        b.iter(|| state.stochastic_measure(&indices, 0.0));
    }

    #[bench]
    fn bench_identity_deep_single_amp(b: &mut Bencher) {
        let n = 6;
        let mut state = FeynmanState::<f64>::new(n);

        for _ in 0..3 {
            let mult = (1.0 / 2.0f64).sqrt();
            let mat = from_reals(&vec![mult, mult, mult, -mult]);
            for i in 0..n {
                let op = make_matrix_op(vec![i], mat.clone()).unwrap();
                state.apply_op(&op);
            }
        }
        let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
        let op = make_matrix_op(vec![0], mat.clone()).unwrap();
        state.apply_op(&op);

        b.iter(|| state.calculate_amplitude(0));
    }
}
