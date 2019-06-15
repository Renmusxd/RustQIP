extern crate num;
extern crate rand;
extern crate rayon;

use std::cmp::{max, min};

use num::complex::Complex;
use rand::prelude::*;
use rayon::prelude::*;

/// Get total magnitude of state.
fn prob_magnitude(input: &Vec<Complex<f64>>) -> f64 {
    input.par_iter().map(Complex::<f64>::norm_sqr).sum()
}

/// Calculate the probability of a given measurement. `measured` gives the bits (as a u64) which has
/// been measured from the qubits at `indices` in the order supplied by `indices`. `input` gives the
/// state from which to measure, representing a total of `n` qubits. And `input_offset` gives the
/// actual index of the lowest indexed entry in `input` in case it's split across multiple vectors
/// (as for distributed computation)
///
/// # Examples
///
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
/// // Make the state |10>, index 0 is always |0> and index 1 is always |1>
/// // test that 0,1 is always |2> (or in other words |1,0>)
/// let input = from_reals(&vec![0.0, 0.0, 1.0, 0.0]);
/// let p = measure_prob(2, 2, &vec![0, 1], &input, 0);
/// assert_eq!(p, 1.0);
/// ```
/// ```
/// use qip::state_ops::from_reals;
/// use qip::measurement_ops::measure_prob;
/// // Make the state |10>, index 0 is always |0> and index 1 is always |1>
/// // test that 1,0 is always |1> (or in other words |0,1>)
/// let input = from_reals(&vec![0.0, 0.0, 1.0, 0.0]);
/// let p = measure_prob(2, 1, &vec![1, 0], &input, 0);
/// assert_eq!(p, 1.0);
/// ```
pub fn measure_prob(n: u64, measured: u64, indices: &Vec<u64>, input: &Vec<Complex<f64>>, input_offset: u64) -> f64 {
    let template: u64 = indices.iter().cloned().enumerate().map(|(i, index)| -> u64 {
        let sel_bit = (measured >> i as u64) & 1;
        sel_bit << index
    }).sum();
    let remaining_indices: Vec<u64> = (0..n).filter(|i| !indices.contains(i)).collect();

    // No parallel iterators available for u64 ranges.
    (0..1 << n - indices.len() as u64).map(|remaining_index_bits: u64| -> f64 {
        let tmp_index: u64 = remaining_indices.iter().clone().enumerate().map(|item| -> u64 {
            let (i, index) = item;
            let sel_bit = (remaining_index_bits >> i as u64) & 1;
            sel_bit << index
        }).sum();
        let index = tmp_index + template;
        if index < input_offset {
            return 0.0;
        }
        let index = index - input_offset;
        if index >= input.len() as u64 {
            return 0.0;
        }
        input[index as usize].norm_sqr()
    }).sum()
}

/// Sample a measurement from a state `input`.
/// Sample from qubits at `indices` and return bits as u64 in order given by `indices`. See
/// `measure_prob` for details.
fn soft_measure(n: u64, indices: &Vec<u64>, input: &Vec<Complex<f64>>, input_offset: u64) -> u64 {
    let mut r = rand::random::<f64>() *
        if input.len() < (1 << n) as usize {
            prob_magnitude(input)
        } else {
            1.0
        };
    let mut measured_indx = 0;
    for (i, c) in input.iter().enumerate() {
        r -= c.norm_sqr();
        if r <= 0.0 {
            measured_indx = i as u64;
            break;
        }
    }
    indices.iter().cloned().enumerate().map(|(i, index)| -> u64 {
        let sel_bit = (measured_indx >> index) & 1;
        sel_bit << i as u64
    }).sum()
}

/// Selects a measured state from `input`, then calls `measure_state` to manipulate the output.
/// Returns the measured state and probability.
pub fn measure(n: u64, indices: &Vec<u64>,
               input: &Vec<Complex<f64>>, output: &mut Vec<Complex<f64>>,
               input_offset: u64, output_offset: u64) -> (u64, f64) {
    let m = soft_measure(n, indices, input, input_offset);
    let p = measure_prob(n, m, indices, input, input_offset);
    measure_state(n, indices, m, p, input, output, input_offset, output_offset);
    (m, p)
}

/// Normalize the output state such that it matches only states which produce the `measured`
/// result and has the same magnitude.
/// This is done by zeroing out the states which cannot give `measured`, and dividing the remaining
/// by the sqrt(1/p) for p=`measured_prob`. See `measure_prob` for details.
pub fn measure_state(n: u64, indices: &Vec<u64>, measured: u64, measured_prob: f64,
                     input: &Vec<Complex<f64>>, output: &mut Vec<Complex<f64>>,
                     input_offset: u64, output_offset: u64) {
    let p = measured_prob;
    if p != 0.0 {
        let p_mult = (1.0 / p).sqrt();

        let row_mask: u64 = indices.iter().map(|index| {
            1 << *index
        }).sum();
        let measured_mask: u64 = indices.iter().enumerate().map(|(i, index)| {
            let sel_bit = (measured >> i as u64) & 1;
            sel_bit << index
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
                    *output = Complex::<f64> {
                        re: 0.0,
                        im: 0.0,
                    }
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
        let input = from_reals(&vec![0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &vec![0], &input, 0);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &vec![0], m, p, &input, &mut output, 0, 0);

        let half: f64 = 1.0 / 2.0;
        assert_eq!(output, from_reals(&vec![half.sqrt(), 0.0, half.sqrt(), 0.0]));
    }

    #[test]
    fn test_measure_state2() {
        let n = 2;
        let m = 1;
        let input = from_reals(&vec![0.5, 0.5, 0.5, 0.5]);
        let p = measure_prob(n, m, &vec![0], &input, 0);
        assert_eq!(p, 0.5);

        let mut output = input.clone();
        measure_state(n, &vec![0], m, p, &input, &mut output, 0, 0);

        let half: f64 = 1.0 / 2.0;
        assert_eq!(output, from_reals(&vec![0.0, half.sqrt(), 0.0, half.sqrt()]));
    }
}