use std::error::Error;
use std::fmt;

/// An error indicating an invalid value/argument was provided.
#[derive(Debug)]
pub struct CircuitError {
    message: String,
}

impl CircuitError {
    /// Make a new CircuitError with a given message.
    pub fn new(message: String) -> Self {
        CircuitError { message }
    }

    /// Make a new CircuitError Err.
    pub fn make_err<T>(message: String) -> Result<T, CircuitError> {
        Err(Self::new(message))
    }

    /// Make a new CircuitError Err.
    pub fn make_str_err<T>(message: &str) -> Result<T, CircuitError> {
        Err(Self::new(message.to_string()))
    }
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for CircuitError {}
