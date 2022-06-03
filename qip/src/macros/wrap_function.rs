#[macro_export]
macro_rules! wrap_fn {
    (@names () <- $name:ident) => {
        $name
    };
    (@names ($($body:tt)*) <- $name:ident) => {
        ($($body)* $name)
    };
    (@names ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        wrap_fn!(@names ($($body)* $name,) <- $($tail)*)
    };
    (@invoke($func:expr, $builder:expr) ($($body:tt)*) <- $name:ident) => {
        $func($builder, $($body)* $name)
    };
    (@invoke($func:expr, $builder:expr) ($($body:tt)*) <- $name:ident, $($tail:tt)*) => {
        wrap_fn!(@invoke($func, $builder) ($($body)* $name,) <- $($tail)*)
    };
    (@unwrap_regs($func:expr, $rs:ident) $name:ident) => {
        let $name = $rs.pop().ok_or_else(|| $crate::errors::CircuitError::new(format!("Error unwrapping {} for {}", stringify!($name), stringify!($func))))?;
    };
    (@unwrap_regs($func:expr, $rs:ident) $name:ident, $($tail:tt)*) => {
        wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
        let $name = $rs.pop().ok_or_else(|| $crate::errors::CircuitError::new(format!("Error unwrapping {} for {}", stringify!($name), stringify!($func))))?;
    };
    (@result_body($builder:expr, $func:expr, $rs:ident, $arg:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*, $arg) ?;
            let $rs: Vec<_> = vec![$($tail)*];
            Ok($rs)
        }
    };
    (@raw_body($builder:expr, $func:expr, $rs:ident, $arg:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*, $arg);
            let $rs: Vec<_> = vec![$($tail)*];
            Ok($rs)
        }
    };
    (@result_body($builder:expr, $func:expr, $rs:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*) ?;
            let $rs: Vec<_> = vec![$($tail)*];
            Ok($rs)
        }
    };
    (@raw_body($builder:expr, $func:expr, $rs:ident) $($tail:tt)*) => {
        {
            wrap_fn!(@unwrap_regs($func, $rs) $($tail)*);
            let wrap_fn!(@names () <- $($tail)*) = wrap_fn!(@invoke($func, $builder) () <- $($tail)*);
            let $rs: Vec<_> = vec![$($tail)*];
            Ok($rs)
        }
    };
    (pub $newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), ($func:expr), $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<CB:$($typetail)*>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    (pub $newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $func:expr, $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<CB:$($typetail)*>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    ($newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), ($func:expr), $($tail:tt)*) => {
        fn $newfunc<CB:$($typetail)*>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    ($newfunc:ident[$($typetail:tt)*]($arg:ident: $argtype:ident), $func:expr, $($tail:tt)*) => {
        fn $newfunc<CB:$($typetail)*>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    (pub $newfunc:ident($arg:ident: $argtype:ident), ($func:expr), $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    (pub $newfunc:ident($arg:ident: $argtype:ident), $func:expr, $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    ($newfunc:ident($arg:ident: $argtype:ident), ($func:expr), $($tail:tt)*) => {
        fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    ($newfunc:ident($arg:ident: $argtype:ident), $func:expr, $($tail:tt)*) => {
        fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>, $arg: $argtype) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs, $arg) $($tail)*)
        }
    };
    (pub $newfunc:ident, ($func:expr), $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs) $($tail)*)
        }
    };
    (pub $newfunc:ident, $func:expr, $($tail:tt)*) => {
        /// Wrapped version of function
        pub fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs) $($tail)*)
        }
    };
    ($newfunc:ident, ($func:expr), $($tail:tt)*) => {
        fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@result_body(b, $func, rs) $($tail)*)
        }
    };
    ($newfunc:ident, $func:expr, $($tail:tt)*) => {
        fn $newfunc<P: Precision, CB: $crate::macros::RecursiveCircuitBuilder<P>>(b: &mut CB, mut rs: Vec<CB::Register>) -> Result<Vec<CB::Register>, $crate::errors::CircuitError> {
            wrap_fn!(@raw_body(b, $func, rs) $($tail)*)
        }
    };
}
