#[cfg(feature = "parallel")]
pub(crate) use rayon::prelude::*;

use crate::utils::extract_bits;
use crate::{Complex, Precision};
use num_traits::Zero;
use qip_iterators::{into_iter, iter, iter_mut};
use std::cmp::{max, min};

/// Get total magnitude of state.
pub fn prob_magnitude<P: Precision>(input: &[Complex<P>]) -> P {
    iter!(input).map(Complex::<P>::norm_sqr).sum()
}

/// Calculate the probability of a given measurement. `measured` gives the bits (as a usize) which has
/// been measured from the qubits at `indices` in the order supplied by `indices`. `input` gives the
/// state from which to measure, representing a total of `n` qubits. And `input_offset` gives the
/// actual index of the lowest indexed entry in `input` in case it's split across multiple vectors
/// (as for distributed computation)
///
/// Keep in mind that qubits are big-endian to match kron product standards.
/// `|abc>` means `q0=a`, `q1=b`, `q2=c`
///
/// # Examples
/// ```
/// use qip::state_ops::matrix_ops::from_reals;
/// use qip::state_ops::measurement_ops::measure_prob;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let p = measure_prob(2, 0, &[0], &input, None);
/// assert_eq!(p, 0.0);
///
/// let p = measure_prob(2, 1, &[0], &input, None);
/// assert_eq!(p, 1.0);
///
/// let p = measure_prob(2, 1, &[0, 1], &input, None);
/// assert_eq!(p, 1.0);
///
/// let p = measure_prob(2, 2, &[1, 0], &input, None);
/// assert_eq!(p, 1.0);
/// ```
pub fn measure_prob<P: Precision>(
    n: usize,
    measured: usize,
    indices: &[usize],
    input: &[Complex<P>],
    input_offset: Option<usize>,
) -> P {
    measure_prob_fn(n, measured, indices, input_offset, |index| {
        if index >= input.len() {
            Complex::zero()
        } else {
            input[index]
        }
    })
}

/// Calculate the probability of a given measurement. `measured` gives the bits (as a usize) which has
/// been measured from the qubits at `indices` in the order supplied by `indices`.
/// `input_offset` gives the actual index of the lowest indexed entry in `input` in case it's split
/// across multiple vectors (as for distributed computation). `input_length` is the length of the
/// number of allowed indexible
pub fn measure_prob_fn<F, P: Precision>(
    n: usize,
    measured: usize,
    indices: &[usize],
    input_offset: Option<usize>,
    f: F,
) -> P
where
    F: Fn(usize) -> Complex<P> + Send + Sync,
{
    let input_offset = input_offset.unwrap_or(0);
    let template: usize = indices
        .iter()
        .cloned()
        .enumerate()
        .fold(0, |acc, (i, index)| -> usize {
            let sel_bit = (measured >> i) & 1;
            acc | (sel_bit << (n - 1 - index))
        });
    let remaining_indices: Vec<usize> = (0..n).filter(|i| !indices.contains(i)).collect();

    let f = |remaining_index_bits: usize| -> Option<P> {
        let tmp_index: usize =
            remaining_indices
                .iter()
                .cloned()
                .enumerate()
                .fold(0, |acc, (i, index)| -> usize {
                    let sel_bit = (remaining_index_bits >> i) & 1;
                    acc | (sel_bit << (n - 1 - index))
                });
        let index = tmp_index + template;
        if index < input_offset {
            None
        } else {
            let index = index - input_offset;
            let amp = f(index);
            if amp == Complex::zero() {
                None
            } else {
                Some(amp.norm_sqr())
            }
        }
    };

    let r = 0usize..1 << remaining_indices.len();
    into_iter!(r).filter_map(f).sum()
}

/// Get probability for each possible measurement of `indices` on `input`.
pub fn measure_probs<P: Precision>(
    n: usize,
    indices: &[usize],
    input: &[Complex<P>],
    input_offset: Option<usize>,
) -> Vec<P> {
    // If there aren't many indices, put the parallelism on the larger list inside measure_prob.
    // Otherwise use parallelism on the super iteration.
    let r = 0usize..1 << indices.len();
    into_iter!(r)
        .map(|measured| measure_prob(n, measured, indices, input, input_offset))
        .collect()
}

/// Sample a measurement from a state `input`. b
/// Sample from qubits at `indices` and return bits  in order given by `indices`. See
/// `measure_prob` for details.
///
/// Keep in mind that qubits are big-endian to match kron product standards.
/// `|abc>` means `q0=a`, `q1=b`, `q2=c`
///
/// # Examples
/// ```
/// use qip::state_ops::matrix_ops::from_reals;
/// use qip::state_ops::measurement_ops::soft_measure;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let m = soft_measure(2, &[0], &input, None);
/// assert_eq!(m, 1);
/// let m = soft_measure(2, &[1], &input, None);
/// assert_eq!(m, 0);
/// let m = soft_measure(2, &[0, 1], &input, None);
/// assert_eq!(m, 0b01);
/// let m = soft_measure(2, &[1, 0], &input, None);
/// assert_eq!(m, 0b10);
/// ```
pub fn soft_measure<P: Precision>(
    n: usize,
    indices: &[usize],
    input: &[Complex<P>],
    input_offset: Option<usize>,
) -> usize {
    let input_offset = input_offset.unwrap_or(0);
    let mut r = P::from(rand::random::<f64>()).unwrap()
        * if input.len() < (1 << n) {
            prob_magnitude(input)
        } else {
            P::one()
        };
    let mut measured_indx = 0;
    for (i, c) in input.iter().enumerate() {
        r -= c.norm_sqr();
        if r <= P::zero() {
            measured_indx = i + input_offset;
            break;
        }
    }
    let indices: Vec<_> = indices.iter().map(|indx| n - 1 - indx).collect();
    extract_bits(measured_indx, &indices)
}

/// A set of measured results we want to receive (used to avoid the randomness of measurement if
/// a given result is desired).
#[derive(Debug)]
pub struct MeasuredCondition<P: Precision> {
    /// Value which was measured
    pub measured: usize,
    /// Chance of having received that value if known.
    pub prob: Option<P>,
}

/// Selects a measured state from `input`, then calls `measure_state` to manipulate the output.
/// Returns the measured state and probability.
pub fn measure<P: Precision>(
    n: usize,
    indices: &[usize],
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    offsets: Option<(usize, usize)>,
    measured: Option<MeasuredCondition<P>>,
) -> (usize, P) {
    let input_offset = offsets.map(|(i, _)| i);
    let m = if let Some(measured) = &measured {
        measured.measured
    } else {
        soft_measure(n, indices, input, input_offset)
    };

    let p = if let Some(measured_prob) = measured.and_then(|m| m.prob) {
        measured_prob
    } else {
        measure_prob(n, m, indices, input, input_offset)
    };
    let measured = (m, p);

    measure_state(n, indices, measured, input, output, offsets);
    measured
}

/// Normalize the output state such that it matches only states which produce the `measured`
/// result and has the same magnitude.
/// This is done by zeroing out the states which cannot give `measured`, and dividing the remaining
/// by the `sqrt(1/p)` for p=`measured_prob`. See `measure_prob` for details.
pub fn measure_state<P: Precision>(
    n: usize,
    indices: &[usize],
    measured: (usize, P),
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    offsets: Option<(usize, usize)>,
) {
    let (measured, measured_prob) = measured;
    let (input_offset, output_offset) = offsets.unwrap_or((0, 0));
    if !measured_prob.is_zero() {
        let p_mult = P::one() / measured_prob.sqrt();

        let row_mask: usize = indices.iter().map(|index| 1 << (n - 1 - index)).sum();
        let measured_mask: usize = indices
            .iter()
            .enumerate()
            .map(|(i, index)| {
                let sel_bit = (measured >> i) & 1;
                sel_bit << (n - 1 - index)
            })
            .sum();

        let lower = max(input_offset, output_offset);
        let upper = min(input_offset + input.len(), output_offset + output.len());
        let input_lower = lower - input_offset;
        let input_upper = upper - input_offset;
        let output_lower = lower - output_offset;
        let output_upper = upper - output_offset;

        let f = |(i, (input, output)): (usize, (&Complex<P>, &mut Complex<P>))| {
            // Calculate the actual row we are on:
            let row = i + lower;
            // Select the bits we are measuring.
            let measured_bits = row & row_mask;
            // Is there a difference between them and the actually measured value?
            if (measured_bits ^ measured_mask) != 0 {
                // This is not a valid measurement, zero out the entry.
                *output = Complex::default();
            } else {
                // Scale the entry.
                *output = (*input) * p_mult;
            }
        };

        let input_iter = iter!(input[input_lower..input_upper]);
        let output_iter = iter_mut!(output[output_lower..output_upper]);
        input_iter.zip(output_iter).enumerate().for_each(f);
    }
}

#[cfg(test)]
mod measurement_tests {
    use super::*;
    use crate::state_ops::matrix_ops::from_reals;

    fn round(c: Complex<f64>) -> Complex<f64> {
        Complex {
            re: c.re.round(),
            im: c.im.round(),
        }
    }

    fn approx_eq(a: &[Complex<f64>], b: &[Complex<f64>], prec: i32) {
        let prec = 10.0f64.powi(-prec);
        let a: Vec<Complex<f64>> = a.iter().map(|f| round(f * prec) / prec).collect();
        let b: Vec<Complex<f64>> = b.iter().map(|f| round(f * prec) / prec).collect();
        assert_eq!(a, b)
    }

    #[test]
    fn test_measure_state() {
        let n = 2;
        let m = 0;
        let input = from_reals(&[0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &[0], &input, None);
        assert!((p - 0.5f64).abs() < f64::EPSILON);

        let mut output = input.clone();
        measure_state(n, &[0], (m, p), &input, &mut output, None);

        let half: f64 = 1.0 / 2.0;
        approx_eq(
            &output,
            &from_reals(&[half.sqrt(), half.sqrt(), 0.0, 0.0]),
            10,
        );
    }

    #[test]
    fn test_measure_state2() {
        let n = 2;
        let m = 1;
        let input = from_reals(&[0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &[0], &input, None);
        assert!((p - 0.5f64).abs() < f64::EPSILON);

        let mut output = input.clone();
        measure_state(n, &[0], (m, p), &input, &mut output, None);

        let half: f64 = 1.0 / 2.0;
        approx_eq(
            &output,
            &from_reals(&[0.0, 0.0, half.sqrt(), half.sqrt()]),
            10,
        );
    }

    #[test]
    fn test_measure_probs() {
        let n = 2;
        let m = 1;
        let input = from_reals(&[0.5, 0.5, 0.5, 0.5]);
        let p = measure_probs(n, &[m], &input, None);
        assert_eq!(p, vec![0.5, 0.5]);
    }
}
