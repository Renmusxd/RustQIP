use std::error::Error;
use std::fmt::{Display, Formatter};

/// An error from building the circuit.
#[derive(Debug)]
pub enum CircuitError {
    /// A generic error.
    Generic(String),
}

impl CircuitError {
    /// Construct a new error.
    pub fn new<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        Self::Generic(msg.into())
    }
}

/// A result which may contain a circuit error.
pub type CircuitResult<T> = Result<T, CircuitError>;

impl Error for CircuitError {}

impl Display for CircuitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generic(msg) => write!(f, "{}", msg),
        }
    }
}
