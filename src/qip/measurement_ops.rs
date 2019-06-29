extern crate num;
extern crate rand;
extern crate rayon;

use std::cmp::{max, min};

use crate::types::Precision;
use num::complex::Complex;
use rand::prelude::*;
use rayon::prelude::*;

/// Get total magnitude of state.
fn prob_magnitude<P: Precision>(input: &[Complex<P>]) -> P {
    input.par_iter().map(Complex::<P>::norm_sqr).sum()
}

/// Calculate the probability of a given measurement. `measured` gives the bits (as a u64) which has
/// been measured from the qubits at `indices` in the order supplied by `indices`. `input` gives the
/// state from which to measure, representing a total of `n` qubits. And `input_offset` gives the
/// actual index of the lowest indexed entry in `input` in case it's split across multiple vectors
/// (as for distributed computation)
///
/// Keep in mind that qubits are big-endian to match kron product standards.
/// |abc> means q0=a, q1=b, q2=c
///
/// # Examples
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let p = measure_prob(2, 0, &[0], &input, 0);
/// assert_eq!(p, 0.0);
/// let p = measure_prob(2, 1, &[0], &input, 0);
/// assert_eq!(p, 1.0);
/// ```
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
/// let p = measure_prob(2, 1, &[0, 1], &input, 0);
/// assert_eq!(p, 1.0);
/// ```
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
/// let p = measure_prob(2, 2, &[1, 0], &input, 0);
/// assert_eq!(p, 1.0);
/// ```
pub fn measure_prob<P: Precision>(n: u64, measured: u64, indices: &[u64], input: &[Complex<P>], input_offset: u64) -> P {
    let template: u64 = indices.iter().cloned().enumerate().map(|(i, index)| -> u64 {
        let sel_bit = (measured >> i as u64) & 1;
        sel_bit << (n - 1 - index)
    }).sum();
    let remaining_indices: Vec<u64> = (0..n).filter(|i| !indices.contains(i)).collect();

    (0u64 .. 1 << (n - indices.len() as u64)).into_par_iter().map(|remaining_index_bits: u64| -> P {
        let tmp_index: u64 = remaining_indices.iter().clone().enumerate().map(|item| -> u64 {
            let (i, index) = item;
            let sel_bit = (remaining_index_bits >> i as u64) & 1;
            sel_bit << (n - 1 - index)
        }).sum();
        let index = tmp_index + template;
        if index < input_offset {
            return P::zero();
        }
        let index = index - input_offset;
        if index >= input.len() as u64 {
            return P::zero();
        }
        input[index as usize].norm_sqr()
    }).sum()
}

/// Sample a measurement from a state `input`. b
/// Sample from qubits at `indices` and return bits as u64 in order given by `indices`. See
/// `measure_prob` for details.
///
/// Keep in mind that qubits are big-endian to match kron product standards.
/// |abc> means q0=a, q1=b, q2=c
/// /// # Examples
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::soft_measure;
///
/// // Make the state |10>, index 0 is always |1> and index 1 is always |0>
/// let input = from_reals(&[0.0, 0.0, 1.0, 0.0]);
///
/// let m = soft_measure(2, &[0], &input, 0);
/// assert_eq!(m, 1);
/// let m = soft_measure(2, &[1], &input, 0);
/// assert_eq!(m, 0);
/// ```
pub fn soft_measure<P: Precision>(n: u64, indices: &[u64], input: &[Complex<P>], input_offset: u64) -> u64 {
    let mut r = P::from(rand::random::<f64>()).unwrap() *
        if input.len() < (1 << n) as usize {
            prob_magnitude(input)
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
    indices.iter().cloned().enumerate().map(|(i, index)| -> u64 {
        let sel_bit = (measured_indx >> (n - 1 - index)) & 1;
        sel_bit << i as u64
    }).sum()
}

/// Selects a measured state from `input`, then calls `measure_state` to manipulate the output.
/// Returns the measured state and probability.
pub fn measure<P: Precision>(n: u64, indices: &[u64],
               input: &[Complex<P>], output: &mut Vec<Complex<P>>,
               input_offset: u64, output_offset: u64) -> (u64, P) {
    let m = soft_measure(n, indices, input, input_offset);
    let p = measure_prob(n, m, indices, input, input_offset);
    measure_state(n, indices, m, p, input, output, input_offset, output_offset);
    (m, p)
}

/// Normalize the output state such that it matches only states which produce the `measured`
/// result and has the same magnitude.
/// This is done by zeroing out the states which cannot give `measured`, and dividing the remaining
/// by the sqrt(1/p) for p=`measured_prob`. See `measure_prob` for details.
pub fn measure_state<P: Precision>(n: u64, indices: &[u64], measured: u64, measured_prob: P,
                     input: &[Complex<P>], output: &mut[Complex<P>],
                     input_offset: u64, output_offset: u64) {
    if !measured_prob.is_zero() {
        let p_mult = P::one() / measured_prob.sqrt();

        let row_mask: u64 = indices.iter().map(|index| {
            1 << (n - 1 - index)
        }).sum();
        let measured_mask: u64 = indices.iter().enumerate().map(|(i, index)| {
            let sel_bit = (measured >> i as u64) & 1;
            sel_bit << (n - 1 - index)
        }).sum();

        let lower = max(input_offset, output_offset);
        let upper = min(input_offset + input.len() as u64,
                        output_offset + output.len() as u64);
        let input_lower = (lower - input_offset) as usize;
        let input_upper = (upper - input_offset) as usize;
        let output_lower = (lower - output_offset) as usize;
        let output_upper = (upper - output_offset) as usize;
        let input_iter = input[input_lower..input_upper].par_iter();
        let output_iter = output[output_lower..output_upper].par_iter_mut();
        input_iter.zip(output_iter).enumerate().for_each(
            |(i, (input, output))| {
                // Calculate the actual row we are on:
                let row = i as u64 + lower;
                if ((row & row_mask) ^ measured_mask) != 0 {
                    // This is not a valid measurement, zero out the entry.
                    *output = Complex::default();
                } else {
                    // Otherwise scale the entry.
                    *output = (*input) * p_mult;
                }
            }
        );
    }
}

#[cfg(test)]
mod measurement_tests {
    use crate::state_ops::from_reals;
    use super::*;

    #[test]
    fn test_measure_state() {
        let n = 2;
        let m = 0;
        let input = from_reals(&[0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &[0], &input, 0);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &[0], m, p, &input, &mut output, 0, 0);

        let half: f64 = 1.0 / 2.0;
        assert_eq!(output, from_reals(&[half.sqrt(), half.sqrt(), 0.0, 0.0]));
    }

    #[test]
    fn test_measure_state2() {
        let n = 2;
        let m = 1;
        let input = from_reals(&[0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &[0], &input, 0);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &[0], m, p, &input, &mut output, 0, 0);

        let half: f64 = 1.0 / 2.0;
        assert_eq!(output, from_reals(&[0.0, 0.0, half.sqrt(), half.sqrt()]));
    }
}