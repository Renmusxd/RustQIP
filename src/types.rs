extern crate num;
extern crate rayon;

use std::iter::Sum;

use num::Float;

/// The float precision of the circuit.
pub trait Precision: Default + Float + Sum + Send + Sync {}

impl Precision for f64 {}
impl Precision for f32 {}
