#![feature(test)]

#[cfg(test)]
mod tests {
    extern crate blas;
    extern crate openblas_src;
    extern crate test;

    use faer_core::mul::matmul;
    use faer_core::{Mat, Parallelism};
    use ndarray::linalg::{general_mat_vec_mul, Dot};
    use ndarray::{Array1, Array2, LinalgScalar};
    use qip_iterators::iterators::MatrixOp;
    use qip_iterators::matrix_ops::{apply_op, apply_ops};
    use sprs::{kronecker_product, CsMat, TriMat};
    use test::Bencher;
    use num_traits::{Num, One, Zero};

    #[bench]
    fn bench_ones_qip(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op = MatrixOp::new_matrix(vec![0], mat);

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| apply_op(n, &op, input, output, 0, 0));
    }

    #[bench]
    fn bench_ones_sprs_reuse(b: &mut Bencher) {
        let n = 12;
        let mut a = TriMat::new((2, 2));
        a.add_triplet(0, 0, 1.0f64);
        a.add_triplet(1, 0, 1.0f64);
        a.add_triplet(0, 1, 1.0f64);
        a.add_triplet(1, 1, 1.0f64);
        let mut acc = a.to_csr::<usize>();

        let eye = CsMat::eye(2);
        for _ in 1..n {
            acc = kronecker_product(acc.view(), eye.view());
        }

        let input = Array1::ones(1 << n);

        b.iter(|| acc.dot(&input));
    }

    #[bench]
    fn bench_ones_sprs_build_each(b: &mut Bencher) {
        let n = 12;

        let input = Array1::ones(1 << n);

        b.iter(|| {
            let mut a = TriMat::new((2, 2));
            a.add_triplet(0, 0, 1.0f64);
            a.add_triplet(1, 0, 1.0f64);
            a.add_triplet(0, 1, 1.0f64);
            a.add_triplet(1, 1, 1.0f64);
            let mut acc = a.to_csr::<usize>();

            let eye = CsMat::eye(2);
            for _ in 1..n {
                acc = kronecker_product(acc.view(), eye.view());
            }
            acc.dot(&input)
        });
    }

    fn make_ones_mat<P: LinalgScalar>(n: usize) -> Array2<P> {
        let mut acc = Array2::from_shape_vec((2, 2), vec![P::one(); 4])
            .expect("Could not make 2x2 matrix.");
        for _ in 1..n {
            acc = ndarray::linalg::kron(&acc, &Array2::eye(2));
        }
        acc
    }

    #[bench]
    fn bench_ones_ndarray_build_each(b: &mut Bencher) {
        let n = 12;
        let input = Array1::ones(1 << n);

        b.iter(|| {
            let acc = make_ones_mat::<f64>(n);
            acc.dot(&input)
        });
    }

    #[bench]
    fn bench_ones_ndarray_reuse(b: &mut Bencher) {
        let n = 12;
        let acc = make_ones_mat::<f64>(n);
        let input = Array1::ones(1 << n);

        b.iter(|| acc.dot(&input));
    }

    #[bench]
    fn bench_ones_faer_reuse_arena_singlethread(b: &mut Bencher) {
        let n = 12;
        let acc = make_ones_mat::<f64>(n);
        let acc = Mat::<f64>::with_dims(1 << n, 1 << n, |i, j| acc[(i, j)]);

        let input = Mat::<f64>::zeros(1 << n, 1);
        let mut output = Mat::<f64>::zeros(1 << n, 1);

        b.iter(|| {
            matmul(
                output.as_mut(),
                acc.as_ref(),
                input.as_ref(),
                None,
                0.0,
                Parallelism::Rayon(1),
            )
        });
    }


    #[bench]
    fn bench_ones_faer_reuse_arena(b: &mut Bencher) {
        // At the time of writing faer doesn't parallelize vector multiplication.
        let n = 12;
        let acc = make_ones_mat::<f64>(n);
        let acc = Mat::<f64>::with_dims(1 << n, 1 << n, |i, j| acc[(i, j)]);

        let input = Mat::<f64>::zeros(1 << n, 1);
        let mut output = Mat::<f64>::zeros(1 << n, 1);

        b.iter(|| {
            matmul(
                output.as_mut(),
                acc.as_ref(),
                input.as_ref(),
                None,
                0.0,
                Parallelism::Rayon(0),
            )
        });
    }

    #[bench]
    fn bench_ones_ndarray_reuse_arena(b: &mut Bencher) {
        let n = 12;
        let acc = make_ones_mat(n);

        let input = Array1::ones(1 << n);
        let mut output = Array1::zeros(1 << n);

        b.iter(|| {
            general_mat_vec_mul(1.0, &acc, &input, 0.0, &mut output);
        });
    }

    #[bench]
    fn bench_large_ones_qip(b: &mut Bencher) {
        let n = 20;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op = MatrixOp::new_matrix(vec![0], mat);

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| apply_op(n, &op, input, output, 0, 0));
    }

    #[bench]
    fn bench_large_ones_sprs_reuse(b: &mut Bencher) {
        let n = 20;
        let mut a = TriMat::new((2, 2));
        a.add_triplet(0, 0, 1.0f64);
        a.add_triplet(1, 0, 1.0f64);
        a.add_triplet(0, 1, 1.0f64);
        a.add_triplet(1, 1, 1.0f64);
        let mut acc = a.to_csr::<usize>();

        let eye = CsMat::eye(2);
        for _ in 1..n {
            acc = kronecker_product(acc.view(), eye.view());
        }

        let input = Array1::ones(1 << n);

        b.iter(|| acc.dot(&input));
    }

    #[bench]
    fn bench_large_ones_sprs_build_each(b: &mut Bencher) {
        let n = 20;

        let input = Array1::ones(1 << n);

        b.iter(|| {
            let mut a = TriMat::new((2, 2));
            a.add_triplet(0, 0, 1.0f64);
            a.add_triplet(1, 0, 1.0f64);
            a.add_triplet(0, 1, 1.0f64);
            a.add_triplet(1, 1, 1.0f64);
            let mut acc = a.to_csr::<usize>();

            let eye = CsMat::eye(2);
            for _ in 1..n {
                acc = kronecker_product(acc.view(), eye.view());
            }
            acc.dot(&input)
        });
    }

    #[bench]
    fn bench_two_qip_series(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op1 = MatrixOp::new_matrix(vec![0], mat);
        let op2 = MatrixOp::new_matrix(vec![1], mat);

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            apply_op(n, &op1, input, output, 0, 0);
            apply_op(n, &op2, input, output, 0, 0);
        });
    }

    #[bench]
    fn bench_two_qip_multi(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op1 = MatrixOp::new_matrix(vec![0], mat);
        let op2 = MatrixOp::new_matrix(vec![1], mat);
        let ops = [op1, op2];

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            apply_ops(n, &ops, input, output, 0, 0);
        });
    }

    #[bench]
    fn bench_three_qip_series(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op1 = MatrixOp::new_matrix(vec![0], mat);
        let op2 = MatrixOp::new_matrix(vec![1], mat);
        let op3 = MatrixOp::new_matrix(vec![2], mat);

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            apply_op(n, &op1, input, output, 0, 0);
            apply_op(n, &op2, input, output, 0, 0);
            apply_op(n, &op3, input, output, 0, 0);
        });
    }

    #[bench]
    fn bench_three_qip_multi(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let op1 = MatrixOp::new_matrix(vec![0], mat);
        let op2 = MatrixOp::new_matrix(vec![1], mat);
        let op3 = MatrixOp::new_matrix(vec![2], mat);
        let ops = [op1, op2, op3];

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            apply_ops(n, &ops, input, output, 0, 0);
        });
    }

    #[bench]
    fn bench_twelve_qip_series(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let ops = (0..n)
            .map(|i| MatrixOp::new_matrix(vec![i], mat))
            .collect::<Vec<_>>();

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            for op in &ops {
                apply_op(n, op, input, output, 0, 0);
            }
        });
    }

    #[bench]
    fn bench_twelve_qip_multi(b: &mut Bencher) {
        let n = 12;

        let mat = [1.0, 1.0, 1.0, 1.0];
        let ops = (0..n)
            .map(|i| MatrixOp::new_matrix(vec![i], mat))
            .collect::<Vec<_>>();

        let arr_input = Array1::ones(1 << n);
        let mut arr_output = Array1::zeros(1 << n);

        let input = arr_input.as_slice().unwrap();
        let output = arr_output.as_slice_mut().unwrap();

        b.iter(|| {
            apply_ops(n, &ops, input, output, 0, 0);
        });
    }
}
