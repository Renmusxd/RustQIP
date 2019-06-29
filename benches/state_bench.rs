#[macro_use]
extern crate bencher;
extern crate qip;
extern crate num;

use bencher::Bencher;

use num::Complex;
use qip::state_ops::*;
use qip::state_ops::QubitOp::*;

/// Make the full op matrix from `ops`.
/// Not very efficient, use only for debugging.
fn make_ops_matrix(n: u64, ops: &Vec<&QubitOp>) -> Vec<Vec<Complex<f64>>> {
    let zeros: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    (0..1 << n).map(|i| {
        let mut input = from_reals(&zeros);
        let output = input.clone();
        input[i] = Complex {
            re: 1.0,
            im: 0.0,
        };
        let (input, _) = ops.iter().fold((input, output), |(input, mut output), op| {
            apply_op(n, op, &input, &mut output, 0, 0, true);
            (output, input)
        });
        input
    }).collect()
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

    let mult = (1.0/2.0f64).sqrt();
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
    let op = make_control_op((0..n - 1).collect(), Matrix(vec![n - 1], mat));

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

    let mult = (1.0/2.0f64).sqrt();
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
    let op = make_control_op((0..n - 1).collect(), Matrix(vec![n - 1], mat));

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity_giant(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let c_indices = (0 .. n - 1).collect();
    let op = make_control_op(c_indices, Matrix(vec![n - 1], mat));

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

fn bench_cidentity_giant_halfprec(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let c_indices = (0 .. n - 1).collect();
    let op = make_control_op(c_indices, Matrix(vec![n - 1], mat));

    let base_vector: Vec<f32> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_op(n, &op, &input, &mut output, 0, 0, false));
}

benchmark_group!(benches, bench_identity, bench_hadamard, bench_cidentity,
                 bench_identity_larger, bench_hadamard_larger, bench_cidentity_larger,
                 bench_cidentity_giant, bench_cidentity_giant_halfprec);
benchmark_main!(benches);