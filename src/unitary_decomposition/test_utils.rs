use crate::unitary_decomposition::decomposition::{BaseUnitary, DecompOp};
use crate::unitary_decomposition::utils::{
    apply_controlled_rotation_and_clean, apply_phase_to_row,
};
use crate::{Complex, Precision};
use num::One;

/// Flatten the sparse matrix and add row information.
pub(crate) fn flat_sparse<T>(v: Vec<Vec<(u64, T)>>) -> Vec<(u64, u64, T)> {
    v.into_iter()
        .enumerate()
        .map(|(row, v)| -> Vec<(u64, u64, T)> {
            v.into_iter()
                .map(|(col, val)| (row as u64, col, val))
                .collect()
        })
        .flatten()
        .collect()
}

/// Print out a sparse matrix.
pub(crate) fn print_sparse<P: Precision>(v: &[Vec<(u64, Complex<P>)>]) {
    v.iter().enumerate().for_each(|(row, v)| {
        print!("{:?}\t", row);
        v.iter().for_each(|(col, val)| {
            let (r, p) = val.to_polar();
            print!("({}, {}||{})\t", col, r, p)
        });
        println!();
    });
}

/// Use the ops and base unitary to reconstruct the decomposed unitary op.
pub(crate) fn reconstruct_unitary<P: Precision + Clone>(
    n: u64,
    ops: &[DecompOp<P>],
    base: &BaseUnitary<P>,
) -> Vec<Vec<(u64, Complex<P>)>> {
    let keep_threshold = P::from(1e-10).unwrap();
    let mut base_mat: Vec<_> = (0..1 << n)
        .map(|indx| {
            if indx == base.top_row {
                vec![(base.top_row, base.dat[0]), (base.bot_row, base.dat[1])]
            } else if indx == base.bot_row {
                vec![(base.top_row, base.dat[2]), (base.bot_row, base.dat[3])]
            } else {
                vec![(indx, Complex::one())]
            }
        })
        .collect();
    base_mat[base.top_row as usize].sort_by_key(|(col, _)| *col);
    base_mat[base.top_row as usize].retain(|(_, val)| val.norm_sqr() >= keep_threshold);
    base_mat[base.bot_row as usize].sort_by_key(|(col, _)| *col);
    base_mat[base.bot_row as usize].retain(|(_, val)| val.norm_sqr() >= keep_threshold);

    ops.iter().for_each(|op| {
        match op {
            DecompOp::Rotation {
                from_bits,
                to_bits,
                theta,
                ..
            } => apply_controlled_rotation_and_clean(
                *from_bits,
                *to_bits,
                *theta,
                &mut base_mat,
                |val| val.norm_sqr() >= keep_threshold,
            ),
            DecompOp::Phase { row, phi } => {
                apply_phase_to_row(*phi, &mut base_mat[(*row) as usize])
            }
            DecompOp::Negate { row_a, row_b, .. } => {
                base_mat.swap((*row_a) as usize, (*row_b) as usize);
            }
        };
    });
    base_mat
}
