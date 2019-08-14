use std::error::Error;
use std::fmt;

/// A cleaner version of the generically boxed type.
pub type BoxError = Box<dyn Error>;

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

    /// Generically box this error.
    pub fn boxme(self) -> BoxError {
        Box::new(self) as Box<dyn Error>
    }

    /// Make a new InvalidValueError Err.
    pub fn make_err<T>(message: String) -> Result<T, InvalidValueError> {
        Err(Self::new(message))
    }

    /// Make a new Boxed InvalidValueError Err.
    pub fn make_boxed_err<T>(message: String) -> Result<T, Box<dyn Error>> {
        Self::make_err(message).map_err(|err| err.boxme())
    }

    /// Make a new InvalidValueError Err.
    pub fn make_str_err<T>(message: &str) -> Result<T, InvalidValueError> {
        Err(Self::new(message.to_string()))
    }

    /// Make a new Boxed InvalidValueError Err.
    pub fn make_str_boxed_err<T>(message: &str) -> Result<T, Box<dyn Error>> {
        Self::make_err(message.to_string()).map_err(|err| err.boxme())
    }
}

impl fmt::Display for InvalidValueError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for InvalidValueError {}
