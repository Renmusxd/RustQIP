use crate::measurement_ops::MeasuredCondition;
use crate::rayon_helper::*;
use crate::state_ops::{full_to_sub, sub_to_full};
use crate::utils::{extract_bits, flip_bits};
use crate::{Complex, Precision};
use std::ops::Add;

pub(crate) fn consolidate_vec<
    K: PartialEq + Ord + Clone + Send + Sync,
    V: Add<Output = V> + Send + Sync,
>(
    mut v: Vec<(K, V)>,
) -> Vec<(K, V)> {
    sort_unstable_by!(v, |(a, _), (b, _)| a.cmp(b));
    v.into_iter().fold(vec![], |mut acc, (indx, val)| {
        let last_indx = acc.last().map(|(indx, _)| indx);
        match last_indx {
            Some(last_indx) if indx.eq(last_indx) => {
                let (last_indx, last_val) = acc.pop().unwrap();
                acc.push((last_indx, last_val + val));
            }
            _ => acc.push((indx, val)),
        }
        acc
    })
}

pub(crate) fn sparse_prob_magnitude<P: Precision>(state: &[(u64, Complex<P>)]) -> P {
    let p: P = iter!(state).map(|(_, v)| v.norm_sqr()).sum();
    p.sqrt()
}

fn sparse_measure_state<P: Precision>(
    n: u64,
    indices: &[u64],
    measured: (u64, P),
    state: Vec<(u64, Complex<P>)>,
) -> Vec<(u64, Complex<P>)> {
    let (m, measured_prob) = measured;
    let p_mult = P::one() / measured_prob.sqrt();
    let rev_m = flip_bits(indices.len(), m);
    let mask = sub_to_full(n, indices, std::u64::MAX, 0);
    let f = |(indx, v): (u64, Complex<P>)| -> Option<(u64, Complex<P>)> {
        if full_to_sub(n, indices, indx & mask) == rev_m {
            Some((indx, v * p_mult))
        } else {
            None
        }
    };

    into_iter!(state).filter_map(f).collect()
}

pub(crate) type MeasurementAndNewState<P> = ((u64, P), Vec<(u64, Complex<P>)>);
pub(crate) fn sparse_measure<P: Precision>(
    n: u64,
    indices: &[u64],
    state: Vec<(u64, Complex<P>)>,
    measured: Option<MeasuredCondition<P>>,
) -> MeasurementAndNewState<P> {
    let m = if let Some(measured) = &measured {
        measured.measured
    } else {
        sparse_soft_measure(n, indices, &state)
    };

    let p = if let Some(measured_prob) = measured.and_then(|m| m.prob) {
        measured_prob
    } else {
        sparse_measure_prob(n, m, indices, &state)
    };
    let measured = (m, p);

    let state = sparse_measure_state(n, indices, measured, state);
    (measured, state)
}

pub(crate) fn sparse_soft_measure<P: Precision>(
    n: u64,
    indices: &[u64],
    state: &[(u64, Complex<P>)],
) -> u64 {
    let mut r = P::from(rand::random::<f64>()).unwrap() * sparse_prob_magnitude(state);
    let mut measured_indx = 0;
    for (i, c) in state.iter() {
        r = r - c.norm_sqr();
        if r <= P::zero() {
            measured_indx = *i;
            break;
        }
    }
    let indices: Vec<_> = indices.iter().map(|indx| n - 1 - indx).collect();
    extract_bits(measured_indx, &indices)
}

pub(crate) fn sparse_measure_prob<P: Precision>(
    n: u64,
    m: u64,
    indices: &[u64],
    state: &[(u64, Complex<P>)],
) -> P {
    let mask = sub_to_full(n, indices, std::u64::MAX, 0);
    let rev_m = flip_bits(indices.len(), m);
    let f = |(indx, v): &(u64, Complex<P>)| -> Option<P> {
        if full_to_sub(n, indices, indx & mask) == rev_m {
            Some(v.norm_sqr())
        } else {
            None
        }
    };

    iter!(state).filter_map(f).sum()
}

pub(crate) fn sparse_measure_probs<P: Precision>(
    n: u64,
    indices: &[u64],
    state: &[(u64, Complex<P>)],
) -> Vec<P> {
    let r = 0u64..1 << indices.len();
    let f = |m: u64| -> P { sparse_measure_prob(n, m, indices, state) };

    into_iter!(r).map(f).collect()
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use crate::state_ops::from_reals;

    fn round(c: Complex<f64>) -> Complex<f64> {
        Complex {
            re: c.re.round(),
            im: c.im.round(),
        }
    }

    fn approx_eq(a: &[(u64, Complex<f64>)], b: &[(u64, Complex<f64>)], prec: i32) {
        let prec = 10.0f64.powi(-prec);
        let a: Vec<_> = a
            .iter()
            .map(|(indx, v)| (indx, round(v * prec) / prec))
            .collect();
        let b: Vec<_> = b
            .iter()
            .map(|(indx, v)| (indx, round(v * prec) / prec))
            .collect();
        assert_eq!(a, b)
    }

    fn make_state<P: Precision>(indices: &[u64], reals: &[P]) -> Vec<(u64, Complex<P>)> {
        let cs = from_reals(reals);
        indices.iter().cloned().zip(cs.into_iter()).collect()
    }

    #[test]
    fn test_measure_state() {
        let n = 2;
        let m = 0;
        let state = make_state(&[0, 1, 2, 3], &[0.5, 0.5, 0.5, 0.5]);
        let p = sparse_measure_prob(n, m, &[0], &state);
        assert!((p - 0.5f64).abs() < f64::EPSILON);

        let output = sparse_measure_state(n, &[0], (m, p), state);

        let half: f64 = 1.0 / 2.0;
        approx_eq(
            &output,
            &make_state(&[0, 1], &[half.sqrt(), half.sqrt()]),
            10,
        );
    }

    #[test]
    fn test_measure_state2() {
        let n = 2;
        let m = 1;
        let state = make_state(&[0, 1, 2, 3], &[0.5, 0.5, 0.5, 0.5]);
        let p = sparse_measure_prob(n, m, &[0], &state);
        assert!((p - 0.5f64).abs() < f64::EPSILON);

        let output = sparse_measure_state(n, &[0], (m, p), state);

        let half: f64 = 1.0 / 2.0;
        approx_eq(
            &output,
            &make_state(&[2, 3], &[half.sqrt(), half.sqrt()]),
            10,
        );
    }

    #[test]
    fn test_measure_probs() {
        let n = 2;
        let m = 1;
        let state = make_state(&[0, 1, 2, 3], &[0.5, 0.5, 0.5, 0.5]);
        let p = sparse_measure_probs(n, &[m], &state);
        assert_eq!(p, vec![0.5, 0.5]);
    }
}
