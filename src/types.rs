extern crate rayon;

use num::Float;
use std::iter::Sum;
use std::fmt::Debug;

/// The float precision of the circuit.
pub trait Precision: Default + Float + Sum + Send + Sync + Debug {}

impl Precision for f64 {}
impl Precision for f32 {}
