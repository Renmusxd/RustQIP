#[derive(Debug)]
pub struct CircuitError {
    pub msg: String,
}

impl CircuitError {
    pub fn new<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        Self { msg: msg.into() }
    }
}
