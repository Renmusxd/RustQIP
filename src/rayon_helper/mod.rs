#[cfg(feature = "parallelism")]
pub(crate) use rayon::prelude::*;

/// Choose between iter and par_iter
#[cfg(not(feature = "parallelism"))]
#[macro_export]
macro_rules! iter {
    ($e:expr) => {
        $e.iter()
    };
}

/// Choose between iter and par_iter
#[cfg(feature = "parallelism")]
#[macro_export]
macro_rules! iter {
    ($e:expr) => {
        $e.par_iter()
    };
}

/// Choose between iter_mut and par_iter_mut
#[cfg(not(feature = "parallelism"))]
#[macro_export]
macro_rules! iter_mut {
    ($e:expr) => {
        $e.iter_mut()
    };
}

/// Choose between iter_mut and par_iter_mut
#[cfg(feature = "parallelism")]
#[macro_export]
macro_rules! iter_mut {
    ($e:expr) => {
        $e.par_iter_mut()
    };
}

/// Choose between into_iter and into_par_iter
#[cfg(not(feature = "parallelism"))]
#[macro_export]
macro_rules! into_iter {
    ($e:expr) => {
        $e.into_iter()
    };
}

/// Choose between into_iter and into_par_iter
#[cfg(feature = "parallelism")]
#[macro_export]
macro_rules! into_iter {
    ($e:expr) => {
        $e.into_par_iter()
    };
}

/// Choose between sort_by_key and par_sort_by_key
#[cfg(not(feature = "parallelism"))]
#[macro_export]
macro_rules! sort_by_key {
    ($e:expr, $f:expr) => {
        $e.sort_by_key($f)
    };
}

/// Choose between sort_by_key and par_sort_by_key
#[cfg(feature = "parallelism")]
#[macro_export]
macro_rules! sort_by_key {
    ($e:expr, $f:expr) => {
        $e.par_sort_by_key($f)
    };
}

/// Choose between sort_unstable_by_key and par_sort_unstable_by_key
#[cfg(not(feature = "parallelism"))]
#[macro_export]
macro_rules! sort_unstable_by {
    ($e:expr, $f:expr) => {
        $e.sort_unstable_by($f)
    };
}

/// Choose between sort_by_key and par_sort_by_key
#[cfg(feature = "parallelism")]
#[macro_export]
macro_rules! sort_unstable_by {
    ($e:expr, $f:expr) => {
        $e.par_sort_unstable_by($f)
    };
}
