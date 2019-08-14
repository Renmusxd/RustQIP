use std::error::Error;
use std::fmt;

/// An error indicating an invalid value/argument was provided.
#[derive(Debug)]
pub struct InvalidValueError {
    message: String,
}

impl InvalidValueError {
    /// Make a new InvalidValueError with a given message.
    pub fn new(message: String) -> Self {
        InvalidValueError { message }
    }

    /// Make a new InvalidValueError Err.
    pub fn make_err<T>(message: String) -> Result<T, InvalidValueError> {
        Err(Self::new(message))
    }

    /// Make a new InvalidValueError Err.
    pub fn make_str_err<T>(message: &str) -> Result<T, InvalidValueError> {
        Err(Self::new(message.to_string()))
    }
}

impl fmt::Display for InvalidValueError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for InvalidValueError {}
