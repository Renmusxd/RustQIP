extern crate num;
use super::pipeline::*;
use num::complex::Complex;

pub fn from_reals(data: &[f64]) -> Vec<Complex<f64>> {
    data.into_iter().map(|x| Complex::<f64> {
        re: x.clone(),
        im: 0.0
    }).collect()
}

pub fn from_tuples(data: &[(f64,f64)]) -> Vec<Complex<f64>> {
    data.into_iter().map(|x| -> Complex<f64> {
        let (r, i) = x;
        Complex::<f64> {
            re: r.clone(),
            im: i.clone()
        }
    }).collect()
}