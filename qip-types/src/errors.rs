/// An error from building the circuit.
#[derive(Debug)]
pub struct CircuitError {
    /// The error message.
    pub msg: String,
}

impl CircuitError {
    /// Construct a new error.
    pub fn new<S>(msg: S) -> Self
        where
            S: Into<String>,
    {
        Self { msg: msg.into() }
    }
}

/// A result which may contain a circuit error.
pub type CircuitResult<T> = Result<T, CircuitError>;
