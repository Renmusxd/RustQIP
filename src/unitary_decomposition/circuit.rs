use super::decomposition::decompose_unitary;
use crate::{Complex, Precision, Qubit, UnitaryBuilder};
use crate::unitary_decomposition::decomposition::DecompOp;
use num::{Zero, One};

/// TODO
pub fn convert_sparse_to_circuit(
    b: &mut dyn UnitaryBuilder,
    q: Qubit,
    sparse_unitary: Vec<Vec<(u64, Complex<f64>)>>,
    drop_below: f64,
) -> Result<Qubit, &'static str> {
    let decomposition = decompose_unitary(q.n(), sparse_unitary, drop_below)?;
    let (ops, base) = decomposition.map_err(|_| "Decomposition failed.")?;

    let qs = b.split_all(q);

    // Clear the correct index if it happens to be set
    let base_mask = base.top_row;
    let qs = negate_difference(b, qs, !0, base_mask);
    let qs = apply_to_index_with_control(b, qs, base.bit_index, |cb, q| {
        cb.mat("Base", q, base.dat.to_vec()).unwrap()
    });
    let mask = base_mask;

    let (qs, mask) = ops.into_iter()
        .fold((qs, base_mask), |(qs, mask), op|{
            match op {
                DecompOp::Phase {row, phi} => {
                    let qs = negate_difference(b, qs, mask, row);
                    // We can apply to any qubit.
                    let qs = apply_to_index_with_control(b, qs, 0, |cb, q| {
                        let phase = Complex {
                            re: 0.0,
                            im: phi
                        }.exp();
                        let phase_mat = vec![Complex::one(), Complex::zero(), Complex::zero(), phase];

                        let name = format!("Phase({:?})", phi);
                        cb.mat(&name, q, phase_mat).unwrap()
                    });
                    (qs, row)
                }
                DecompOp::Rotation {from_bits, to_bits, bit_index, theta} => {
                    let new_mask = from_bits; // TODO from_bits?
                    let qs = negate_difference(b, qs, mask, new_mask);
                    let qs = apply_to_index_with_control(b, qs, bit_index, |cb, q| {
                        let (s, c) = theta.sin_cos();
                        let name = format!("Rotate({:?})", theta);
                        cb.real_mat(&name, q, &[c, -s, s, c]).unwrap()
                    });
                    (qs, new_mask)
                }
            }
        });
    let qs = negate_difference(b, qs, mask, !0);
    let q = b.merge(qs);
    Ok(q)
}

fn apply_to_index_with_control<F: Fn(&mut dyn UnitaryBuilder, Qubit) -> Qubit>(
    b: &mut dyn UnitaryBuilder,
    mut qs: Vec<Qubit>,
    indx: u64,
    f: F,
) -> Vec<Qubit> {
    let q = qs.remove(indx as usize);
    let cq = b.merge(qs);
    let mut cb = b.with_condition(cq);
    let q = f(&mut cb, q);
    let cq = cb.release_qubit();
    let mut qs = b.split_all(cq);
    qs.insert(indx as usize, q);
    qs
}

fn negate_difference(
    b: &mut dyn UnitaryBuilder,
    qs: Vec<Qubit>,
    old_mask: u64,
    new_mask: u64,
) -> Vec<Qubit> {
    let needs_negation = old_mask ^ new_mask;
    (0..qs.len() as u64)
        .map(|indx| ((needs_negation >> indx) & 1) == 1)
        .zip(qs.into_iter())
        .map(|(negate, q)|  {
            if negate {
                b.not(q)
            } else {
                q
            }
        })
        .collect()
}


#[cfg(test)]
mod unitary_decomp_circuit_tests {
    use super::*;
    use crate::{OpBuilder, run_debug};
    use crate::unitary_decomposition::utils::flat_sparse;
    use crate::pipeline::make_circuit_matrix;

    const EPSILON: f64 = 0.00000000001;

    fn sparse_from_reals<P: Precision>(v: Vec<Vec<(u64, P)>>) -> Vec<Vec<(u64, Complex<P>)>> {
        let v = v
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|(col, r)| (col, (r, P::zero())))
                    .collect()
            })
            .collect();
        sparse_from_tuples(v)
    }

    fn sparse_from_tuples<P: Precision>(v: Vec<Vec<(u64, (P, P))>>) -> Vec<Vec<(u64, Complex<P>)>> {
        v.into_iter()
            .map(|v| {
                v.into_iter()
                    .map(|(col, (a, b))| (col, Complex { re: a, im: b }))
                    .collect()
            })
            .collect()
    }

    fn flat_round(v: Vec<Vec<(u64, Complex<f64>)>>, prec: i32) -> Vec<(u64, u64, Complex<f64>)> {
        let flat = flat_sparse(v);
        flat.into_iter()
            .map(|(row, col, val)| {
                let p = 10.0f64.powi(prec);
                let val = Complex {
                    re: (val.re * p).round() / p,
                    im: (val.im * p).round() / p,
                };
                (row, col, val)
            })
            .filter(|(_, _, v)| *v != Complex::zero())
            .collect()
    }

    #[test]
    fn test_decompose_basic() -> Result<(), &'static str> {
        let v = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(2, 1.0)],
            vec![(3, 1.0)],
        ];
        let v = sparse_from_reals(v);
        let flat_v = flat_round(v.clone(), 10);

        let n = 2;
        let mut b = OpBuilder::new();
        let q = b.qubit(n).unwrap();

        let q = convert_sparse_to_circuit(&mut b, q, v, EPSILON)?;

        run_debug(&q);

        let reconstructed = make_circuit_matrix::<f64>(n, &q, false);
        let reconstructed = reconstructed.into_iter().map(|v| {
            v.into_iter().enumerate().map(|(indx, c)| (indx as u64, c)).collect()
        }).collect();
        let flat_reconstructed = flat_round(reconstructed, 10);

        assert_eq!(flat_reconstructed, flat_v);

        Ok(())
    }



}