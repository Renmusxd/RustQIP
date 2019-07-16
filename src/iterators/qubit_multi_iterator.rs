use std::marker::PhantomData;
use num::complex::Complex;
use crate::iterators;
use crate::types::Precision;
use crate::utils::*;

/// Iterator which provides the indices of nonzero columns for a given row of a matrix
pub struct MultiOpIterator<'a, 'b, P: Precision> {
    n: u64,
    iterators: &'a [&'b Iterator<Item=(u64, Complex<P>)>],
    last_col: Option<u64>,
}

impl<'a, 'b, P: Precision> MultiOpIterator<'a, 'b, P> {
    pub fn new(row: u64, n: u64, iterators: &'a [&'b Iterator<Item=(u64, Complex<P>)>]) -> MultiOpIterator<'a, 'b, P> {
        MultiOpIterator {
            n,
            iterators,
            last_col: None,
        }
    }
}

impl<'a, 'b, P: Precision> std::iter::Iterator for MultiOpIterator<'a, 'b, P> {
    type Item = (u64, Complex<P>);

    fn next(&mut self) -> Option<Self::Item> {
       unimplemented!()
    }
}