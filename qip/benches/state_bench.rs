#![feature(test)]

use num_complex::Complex;
use qip::state_ops::matrix_ops::from_reals;
use qip_iterators::iterators::MatrixOp;
use qip_iterators::matrix_ops::apply_op;

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
fn make_ops_matrix(n: usize, ops: &[&MatrixOp<Complex<f64>>]) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n)
        .map(|i| {
            let mut input = from_reals(&zeros);
            let output = input.clone();
            input[i] = Complex { re: 1.0, im: 0.0 };
            let (input, _) = ops.iter().fold((input, output), |(input, mut output), op| {
                apply_op(n, op, &input, &mut output, 0, 0);
                (output, input)
            });
            input
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate test;

    use qip::state_ops::matrix_ops::{make_control_op, make_matrix_op};
    use qip_iterators::iterators::MatrixOp::{Matrix, SparseMatrix};
    use qip_iterators::matrix_ops::apply_ops;
    use test::Bencher;

    #[bench]
    fn bench_identity(b: &mut Bencher) {
        let n = 3;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);

        let ops = (0..n)
            .map(|i| Matrix(vec![i], mat.clone()))
            .collect::<Vec<_>>();
        let ops_ref = ops.iter().collect::<Vec<_>>();

        let mat_nested = make_ops_matrix(n, &ops_ref);
        let mat = mat_nested.into_iter().flatten().collect();
        let op = Matrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_hadamard(b: &mut Bencher) {
        let n = 3;

        let mult = (1.0 / 2.0f64).sqrt();
        let mat = from_reals(&[mult, mult, mult, -mult]);

        let ops = (0..n)
            .map(|i| Matrix(vec![i], mat.clone()))
            .collect::<Vec<_>>();
        let ops_ref = ops.iter().collect::<Vec<_>>();

        let mat_nested = make_ops_matrix(n, &ops_ref);
        let mat = mat_nested.into_iter().flatten().collect();
        let op = Matrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_cidentity(b: &mut Bencher) {
        let n = 3;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);
        let op = make_control_op((0..n - 1).collect(), Matrix(vec![n - 1], mat)).unwrap();

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_identity_larger(b: &mut Bencher) {
        let n = 8;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);

        let ops = (0..n)
            .map(|i| Matrix(vec![i], mat.clone()))
            .collect::<Vec<_>>();
        let ops_ref = ops.iter().collect::<Vec<_>>();

        let mat_nested = make_ops_matrix(n, &ops_ref);
        let mat = mat_nested.into_iter().flatten().collect();
        let op = Matrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_hadamard_larger(b: &mut Bencher) {
        let n = 8;

        let mult = (1.0 / 2.0f64).sqrt();
        let mat = from_reals(&[mult, mult, mult, -mult]);

        let ops = (0..n)
            .map(|i| Matrix(vec![i], mat.clone()))
            .collect::<Vec<_>>();
        let ops_ref = ops.iter().collect::<Vec<_>>();

        let mat_nested = make_ops_matrix(n, &ops_ref);
        let mat = mat_nested.into_iter().flatten().collect();
        let op = Matrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_hadamard_larger_single(b: &mut Bencher) {
        let n = 24;

        let mult = (1.0 / 2.0f64).sqrt();
        let mat = from_reals(&[mult, mult, mult, -mult]);

        let op = Matrix(vec![0], mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_cidentity_larger(b: &mut Bencher) {
        let n = 8;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);
        let op = make_matrix_op(vec![n - 1], mat).unwrap();
        let op = make_control_op((0..n - 1).collect(), op).unwrap();

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_cidentity_giant(b: &mut Bencher) {
        let n = 16;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);
        let c_indices = (0..n - 1).collect();
        let op = make_matrix_op(vec![n - 1], mat).unwrap();
        let op = make_control_op(c_indices, op).unwrap();

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_cidentity_giant_halfprec(b: &mut Bencher) {
        let n = 16;

        let mat = from_reals(&[1.0, 0.0, 0.0, 1.0]);
        let c_indices = (0..n - 1).collect();
        let op = make_matrix_op(vec![n - 1], mat).unwrap();
        let op = make_control_op(c_indices, op).unwrap();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_apply_two_swaps_small(b: &mut Bencher) {
        let n = 3;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
        let op2 = make_matrix_op(vec![1], mat).unwrap();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            apply_op(n, &op1, &input, &mut output, 0, 0);
            apply_op(n, &op2, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_apply_two_swaps_small_multiops(b: &mut Bencher) {
        let n = 3;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
        let op2 = make_matrix_op(vec![1], mat).unwrap();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        let ops = [op1, op2];
        b.iter(|| {
            apply_ops(n, &ops, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_apply_two_swaps_large(b: &mut Bencher) {
        let n = 16;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
        let op2 = make_matrix_op(vec![1], mat).unwrap();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            apply_op(n, &op1, &input, &mut output, 0, 0);
            apply_op(n, &op2, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_apply_two_swaps_large_multiops(b: &mut Bencher) {
        let n = 16;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
        let op2 = make_matrix_op(vec![1], mat).unwrap();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        let ops = [op1, op2];
        b.iter(|| {
            apply_ops(n, &ops, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_apply_many_swaps_small(b: &mut Bencher) {
        let n = 5;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let ops: Vec<_> = (0..n)
            .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
            .collect();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            ops.iter()
                .for_each(|op| apply_op(n, op, &input, &mut output, 0, 0))
        });
    }

    #[bench]
    fn bench_apply_many_swaps_small_multiops(b: &mut Bencher) {
        let n = 5;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let ops: Vec<_> = (0..n)
            .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
            .collect();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            apply_ops(n, &ops, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_apply_many_swaps_large(b: &mut Bencher) {
        let n = 10;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let ops: Vec<_> = (0..n)
            .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
            .collect();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            ops.iter()
                .for_each(|op| apply_op(n, op, &input, &mut output, 0, 0))
        });
    }

    #[bench]
    fn bench_apply_many_swaps_large_multiops(b: &mut Bencher) {
        let n = 10;

        let mat = from_reals(&[0.0, 1.0, 1.0, 0.0]);
        let ops: Vec<_> = (0..n)
            .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
            .collect();

        let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| {
            apply_ops(n, &ops, &input, &mut output, 0, 0);
        });
    }

    #[bench]
    fn bench_identity_sparse(b: &mut Bencher) {
        let n = 5;

        let one = Complex::<f64> { re: 1.0, im: 0.0 };
        let mat = (0..1 << n).map(|indx| vec![(indx, one)]).collect();
        let op = SparseMatrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_identity_larger_sparse(b: &mut Bencher) {
        let n = 10;

        let one = Complex::<f64> { re: 1.0, im: 0.0 };
        let mat = (0..1 << n).map(|indx| vec![(indx, one)]).collect();
        let op = SparseMatrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }

    #[bench]
    fn bench_identity_giant_sparse(b: &mut Bencher) {
        let n = 16;

        let one = Complex::<f64> { re: 1.0, im: 0.0 };
        let mat = (0..1 << n).map(|indx| vec![(indx, one)]).collect();
        let op = SparseMatrix((0..n).collect(), mat);

        let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
        let input = from_reals(&base_vector);
        let mut output = from_reals(&base_vector);

        b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0));
    }
}
