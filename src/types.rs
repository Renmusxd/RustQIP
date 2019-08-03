extern crate rayon;

use num::Float;
use std::iter::Sum;

/// The float precision of the circuit.
pub trait Precision: Default + Float + Sum + Send + Sync {}

impl Precision for f64 {}
impl Precision for f32 {}
