#[macro_use]
extern crate bencher;
extern crate qip;

use bencher::Bencher;

use qip::state_ops::*;
use qip::state_ops::QubitOp::*;

fn bench_identity(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);

    let ops: Vec<QubitOp> = (0..n).map(|i| MatrixOp(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_op_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = MatrixOp((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_matrices(n, &vec![&op], &input, &mut output, 0, 0));
}

fn bench_cidentity(b: &mut Bencher) {
    let n = 3;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let op = ControlOp((0..n - 1).collect(), Box::new(MatrixOp(vec![n - 1], mat)));

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_matrices(n, &vec![&op], &input, &mut output, 0, 0));
}

fn bench_identity_larger(b: &mut Bencher) {
    let n = 8;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);

    let ops: Vec<QubitOp> = (0..n).map(|i| MatrixOp(vec![i], mat.clone())).collect();
    let ops_ref = ops.iter().collect();

    let mat_nested = make_op_matrix(n, &ops_ref);
    let mat = mat_nested.into_iter().flatten().collect();
    let op = MatrixOp((0..n).collect(), mat);

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_matrices(n, &vec![&op], &input, &mut output, 0, 0));
}

fn bench_cidentity_larger(b: &mut Bencher) {
    let n = 8;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let op = ControlOp((0..n - 1).collect(), Box::new(MatrixOp(vec![n - 1], mat)));

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_matrices(n, &vec![&op], &input, &mut output, 0, 0));
}

fn bench_cidentity_giant(b: &mut Bencher) {
    let n = 16;

    let mat = from_reals(&vec![1.0, 0.0, 0.0, 1.0]);
    let op = ControlOp(vec![0], Box::new(MatrixOp(vec![n - 1], mat)));

    let base_vector: Vec<f64> = (0..1 << n).map(|_| 0.0).collect();
    let input = from_reals(&base_vector);
    let mut output = from_reals(&base_vector);

    b.iter(|| apply_matrices(n, &vec![&op], &input, &mut output, 0, 0));
}

benchmark_group!(benches, bench_identity, bench_cidentity,
                 bench_identity_larger, bench_cidentity_larger,
                 bench_cidentity_giant);
benchmark_main!(benches);