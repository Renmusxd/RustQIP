use num::Float;
use std::fmt::Display;
use std::iter::{Product, Sum};

/// The float precision of the circuit.
pub trait Precision: Default + Float + Sum + Send + Sync + Display + Product {}

impl Precision for f64 {}
impl Precision for f32 {}
