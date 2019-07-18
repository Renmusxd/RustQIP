#[macro_use]
extern crate bencher;
extern crate num;
extern crate qip;

use bencher::Bencher;

use num::Complex;
use qip::state_ops::QubitOp::*;
use qip::state_ops::*;

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
fn make_ops_matrix(n: u64, ops: &Vec<&QubitOp>) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n)
        .map(|i| {
            let mut input = from_reals(&zeros);
            let output = input.clone();
            input[i] = Complex { re: 1.0, im: 0.0 };
            let (input, _) = ops.iter().fold((input, output), |(input, mut output), op| {
                apply_op(n, op, &input, &mut output, 0, 0, true);
                (output, input)
            });
            input
        })
        .collect()
}

fn bench_identity(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);

    let ops: Vec<QubitOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_ops_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = Matrix((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_hadamard(b: &mut Bencher) {
    let n = 3;

    let mult = (1.0 / 2.0f64).sqrt();
    let mat = from_reals(&vec![mult, mult, mult, -mult]);

    let ops: Vec<QubitOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_ops_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = Matrix((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let op = make_control_op((0..n - 1).collect(), Matrix(vec![n - 1], mat)).unwrap();

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_identity_larger(b: &mut Bencher) {
    let n = 8;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);

    let ops: Vec<QubitOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_ops_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = Matrix((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_hadamard_larger(b: &mut Bencher) {
    let n = 8;

    let mult = (1.0 / 2.0f64).sqrt();
    let mat = from_reals(&vec![mult, mult, mult, -mult]);

    let ops: Vec<QubitOp> = (0..n).map(|i| Matrix(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_ops_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = Matrix((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity_larger(b: &mut Bencher) {
    let n = 8;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let op = make_matrix_op(vec![n - 1], mat).unwrap();
    let op = make_control_op((0..n - 1).collect(), op).unwrap();

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity_giant(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let c_indices = (0..n - 1).collect();
    let op = make_matrix_op(vec![n - 1], mat).unwrap();
    let op = make_control_op(c_indices, op).unwrap();

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity_giant_halfprec(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let c_indices = (0..n - 1).collect();
    let op = make_matrix_op(vec![n - 1], mat).unwrap();
    let op = make_control_op(c_indices, op).unwrap();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_apply_two_swaps_small(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
    let op2 = make_matrix_op(vec![1], mat).unwrap();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_op(n, &op1, &input, &mut output, 0, 0, false);
        apply_op(n, &op2, &input, &mut output, 0, 0, false);
    });
}

fn bench_apply_two_swaps_small_multiops(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
    let op2 = make_matrix_op(vec![1], mat).unwrap();

    let r_op1 = &op1;
    let r_op2 = &op2;

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_ops(n, &[r_op1, r_op2], &input, &mut output, 0, 0, false);
    });
}

fn bench_apply_two_swaps_large(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
    let op2 = make_matrix_op(vec![1], mat).unwrap();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_op(n, &op1, &input, &mut output, 0, 0, false);
        apply_op(n, &op2, &input, &mut output, 0, 0, false);
    });
}

fn bench_apply_two_swaps_large_multiops(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let op1 = make_matrix_op(vec![0], mat.clone()).unwrap();
    let op2 = make_matrix_op(vec![1], mat).unwrap();

    let r_op1 = &op1;
    let r_op2 = &op2;

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_ops(n, &[r_op1, r_op2], &input, &mut output, 0, 0, false);
    });
}

fn bench_apply_many_swaps_small(b: &mut Bencher) {
    let n = 5;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let ops: Vec<_> = (0..n)
        .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
        .collect();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        ops.iter()
            .for_each(|op| apply_op(n, op, &input, &mut output, 0, 0, false))
    });
}

fn bench_apply_many_swaps_small_multiops(b: &mut Bencher) {
    let n = 5;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let ops: Vec<_> = (0..n)
        .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
        .collect();
    let r_ops: Vec<_> = ops.iter().collect();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_ops(n, &r_ops, &input, &mut output, 0, 0, false);
    });
}

fn bench_apply_many_swaps_large(b: &mut Bencher) {
    let n = 10;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let ops: Vec<_> = (0..n)
        .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
        .collect();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        ops.iter()
            .for_each(|op| apply_op(n, op, &input, &mut output, 0, 0, false))
    });
}

fn bench_apply_many_swaps_large_multiops(b: &mut Bencher) {
    let n = 10;

    let mat = from_reals(&vec![0.0, 1.0, 1.0, 0.0]);
    let ops: Vec<_> = (0..n)
        .map(|indx| make_matrix_op(vec![indx], mat.clone()).unwrap())
        .collect();
    let r_ops: Vec<_> = ops.iter().collect();

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| {
        apply_ops(n, &r_ops, &input, &mut output, 0, 0, false);
    });
}


fn bench_identity_larger_sparse(b: &mut Bencher) {
    let n = 8;

    let one = Complex::<f64> { re: 1.0, im: 0.0 };
    let mat = (0 .. 1 << n).map(|indx| vec![(indx, one)]).collect();
    let op = SparseMatrix((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

benchmark_group!(
    benches,
    bench_identity,
    bench_hadamard,
    bench_cidentity,
    bench_identity_larger,
    bench_hadamard_larger,
    bench_cidentity_larger,
    bench_cidentity_giant,
    bench_cidentity_giant_halfprec,
    bench_apply_two_swaps_small,
    bench_apply_two_swaps_small_multiops,
    bench_apply_two_swaps_large,
    bench_apply_two_swaps_large_multiops,
    bench_apply_many_swaps_small,
    bench_apply_many_swaps_small_multiops,
    bench_apply_many_swaps_large,
    bench_apply_many_swaps_large_multiops,
    bench_identity_larger_sparse
);
benchmark_main!(benches);
