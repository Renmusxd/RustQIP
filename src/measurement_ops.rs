extern crate rand;
extern crate rayon;
use crate::{Complex, Precision};
use num::Zero;
use rayon::prelude::*;
use std::cmp::{max, min};

/// Get total magnitude of state.
pub fn prob_magnitude<P: Precision>(input: &[Complex<P>], multithread: bool) -> P {
    if multithread {
        input.par_iter().map(Complex::<P>::norm_sqr).sum()
    } else {
        input.iter().map(Complex::<P>::norm_sqr).sum()
    }
}

/// Calculate the probability of a given measurement. `measured` gives the bits (as a u64) which has
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
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let p = measure_prob(2, 0, &[0], &input, None, false);
/// assert_eq!(p, 0.0);
///
/// let p = measure_prob(2, 1, &[0], &input, None, false);
/// assert_eq!(p, 1.0);
///
/// let p = measure_prob(2, 1, &[0, 1], &input, None, false);
/// assert_eq!(p, 1.0);
///
/// let p = measure_prob(2, 2, &[1, 0], &input, None, false);
/// assert_eq!(p, 1.0);
/// ```
pub fn measure_prob<P: Precision>(
    n: u64,
    measured: u64,
    indices: &[u64],
    input: &[Complex<P>],
    input_offset: Option<u64>,
    multithread: bool,
) -> P {
    let input_offset = input_offset.unwrap_or(0);
    let template: u64 = indices
        .iter()
        .cloned()
        .enumerate()
        .fold(0, |acc, (i, index)| -> u64 {
            let sel_bit = (measured >> i as u64) & 1;
            acc | (sel_bit << (n - 1 - index))
        });
    let remaining_indices: Vec<u64> = (0..n).filter(|i| !indices.contains(i)).collect();

    let f = |remaining_index_bits: u64| -> Option<P> {
        let tmp_index: u64 =
            remaining_indices
                .iter()
                .clone()
                .enumerate()
                .fold(0, |acc, (i, index)| -> u64 {
                    let sel_bit = (remaining_index_bits >> i as u64) & 1;
                    acc | (sel_bit << (n - 1 - index))
                });
        let index = tmp_index + template;
        if index < input_offset {
            None
        } else {
            let index = index - input_offset;
            if index >= input.len() as u64 || input[index as usize] == Complex::zero(){
                None
            } else {
                Some(input[index as usize].norm_sqr())
            }
        }
    };

    let r = 0u64..1 << remaining_indices.len();
    if multithread {
        r.into_par_iter().filter_map(f).sum()
    } else {
        r.filter_map(f).sum()
    }
}

/// Get probability for each possible measurement of `indices` on `input`.
pub fn measure_probs<P: Precision>(
    n: u64,
    indices: &[u64],
    input: &[Complex<P>],
    input_offset: Option<u64>,
    multithread: bool,
) -> Vec<P> {
    // If there aren't many indices, put the parallelism on the larger list inside measure_prob.
    // Otherwise use parallelism on the super iteration.
    let r = 0u64..1 << indices.len();
    if multithread && (indices.len() as u64 > (n >> 1)) {
        r.into_par_iter()
            .map(|measured| measure_prob(n, measured, indices, input, input_offset, false))
            .collect()
    } else {
        r.map(|measured| measure_prob(n, measured, indices, input, input_offset, multithread))
            .collect()
    }
}

/// Sample a measurement from a state `input`. b
/// Sample from qubits at `indices` and return bits as u64 in order given by `indices`. See
/// `measure_prob` for details.
///
/// Keep in mind that qubits are big-endian to match kron product standards.
/// `|abc>` means `q0=a`, `q1=b`, `q2=c`
///
/// # Examples
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::soft_measure;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let m = soft_measure(2, &[0], &input, None, false);
/// assert_eq!(m, 1);
/// let m = soft_measure(2, &[1], &input, None, false);
/// assert_eq!(m, 0);
/// ```
pub fn soft_measure<P: Precision>(
    n: u64,
    indices: &[u64],
    input: &[Complex<P>],
    input_offset: Option<u64>,
    multithread: bool,
) -> u64 {
    let input_offset = input_offset.unwrap_or(0);
    let mut r = P::from(rand::random::<f64>()).unwrap()
        * if input.len() < (1 << n) as usize {
            prob_magnitude(input, multithread)
        } else {
            P::one()
        };
    let mut measured_indx = 0;
    for (i, c) in input.iter().enumerate() {
        r = r - c.norm_sqr();
        if r <= P::zero() {
            measured_indx = i as u64 + input_offset;
            break;
        }
    }
    indices
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, index)| -> u64 {
            let sel_bit = (measured_indx >> (n - 1 - index)) & 1;
            sel_bit << i as u64
        })
        .sum()
}

/// A set of measured results we want to receive (used to avoid the randomness of measurement if
/// a given result is desired).
#[derive(Debug)]
pub struct MeasuredCondition<P: Precision> {
    /// Value which was measured
    pub measured: u64,
    /// Chance of having received that value if known.
    pub prob: Option<P>,
}

/// Selects a measured state from `input`, then calls `measure_state` to manipulate the output.
/// Returns the measured state and probability.
pub fn measure<P: Precision>(
    n: u64,
    indices: &[u64],
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    offsets: Option<(u64, u64)>,
    measured: Option<MeasuredCondition<P>>,
    multithread: bool,
) -> (u64, P) {
    let input_offset = offsets.map(|(i, _)| i);
    let m = if let Some(measured) = &measured {
        measured.measured
    } else {
        soft_measure(n, indices, input, input_offset, multithread)
    };

    let p = if let Some(measured_prob) = measured.and_then(|m| m.prob) {
        measured_prob
    } else {
        measure_prob(n, m, indices, input, input_offset, multithread)
    };
    let measured = (m, p);

    measure_state(n, indices, measured, input, output, offsets, multithread);
    measured
}

/// Normalize the output state such that it matches only states which produce the `measured`
/// result and has the same magnitude.
/// This is done by zeroing out the states which cannot give `measured`, and dividing the remaining
/// by the `sqrt(1/p)` for p=`measured_prob`. See `measure_prob` for details.
pub fn measure_state<P: Precision>(
    n: u64,
    indices: &[u64],
    measured: (u64, P),
    input: &[Complex<P>],
    output: &mut [Complex<P>],
    offsets: Option<(u64, u64)>,
    multithread: bool,
) {
    let (measured, measured_prob) = measured;
    let (input_offset, output_offset) = offsets.unwrap_or((0, 0));
    if !measured_prob.is_zero() {
        let p_mult = P::one() / measured_prob.sqrt();

        let row_mask: u64 = indices.iter().map(|index| 1 << (n - 1 - index)).sum();
        let measured_mask: u64 = indices
            .iter()
            .enumerate()
            .map(|(i, index)| {
                let sel_bit = (measured >> i as u64) & 1;
                sel_bit << (n - 1 - index)
            })
            .sum();

        let lower = max(input_offset, output_offset);
        let upper = min(
            input_offset + input.len() as u64,
            output_offset + output.len() as u64,
        );
        let input_lower = (lower - input_offset) as usize;
        let input_upper = (upper - input_offset) as usize;
        let output_lower = (lower - output_offset) as usize;
        let output_upper = (upper - output_offset) as usize;

        let f = |(i, (input, output)): (usize, (&Complex<P>, &mut Complex<P>))| {
            // Calculate the actual row we are on:
            let row = i as u64 + lower;
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

        if multithread {
            let input_iter = input[input_lower..input_upper].par_iter();
            let output_iter = output[output_lower..output_upper].par_iter_mut();
            input_iter.zip(output_iter).enumerate().for_each(f);
        } else {
            let input_iter = input[input_lower..input_upper].iter();
            let output_iter = output[output_lower..output_upper].iter_mut();
            input_iter.zip(output_iter).enumerate().for_each(f);
        }
    }
}

#[cfg(test)]
mod measurement_tests {
    use super::*;
    use crate::state_ops::from_reals;

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
        let p = measure_prob(n, m, &[0], &input, None, false);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &[0], (m, p), &input, &mut output, None, false);

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
        let p = measure_prob(n, m, &[0], &input, None, false);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &[0], (m, p), &input, &mut output, None, false);

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
        let p = measure_probs(n, &[m], &input, None, false);
        assert_eq!(p, vec![0.5, 0.5]);
    }
}
