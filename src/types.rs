extern crate rayon;

use num::Float;
use std::fmt::Display;
use std::iter::Sum;

/// The float precision of the circuit.
pub trait Precision: Default + Float + Sum + Send + Sync + Display {}

impl Precision for f64 {}
impl Precision for f32 {}
