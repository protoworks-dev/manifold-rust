//! Index-ordered maps, parallel under the `parallel` feature.
//!
//! Each emitter computes element `k` of its output from `k` alone, so the
//! parallel and the sequential build give the same vector.

/// Below this many elements a map stays on the calling thread.
#[cfg(feature = "parallel")]
const PARALLEL_FROM: usize = 4_096;

/// `f(0), f(1), .. f(n - 1)`, in that order.
#[cfg(feature = "parallel")]
pub(crate) fn map_indexed<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    use rayon::prelude::*;
    if n >= PARALLEL_FROM {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// `f(0), f(1), .. f(n - 1)`, in that order.
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_indexed<T, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}
