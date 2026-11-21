//! The one error type of this crate.

use std::fmt;

/// Why a surface could not be read or an output could not be produced.
#[derive(Clone, Debug, PartialEq)]
pub enum EmitError {
    /// A parameter that must be a finite number greater than zero was not.
    NotPositiveFinite {
        /// Which quantity was refused.
        quantity: &'static str,
        /// The value that was given.
        value: f64,
    },
    /// A parameter that must be a finite number of zero or more was not.
    NotFiniteNonNegative {
        /// Which quantity was refused.
        quantity: &'static str,
        /// The value that was given.
        value: f64,
    },
    /// The mesh's buffers do not describe whole vertices or whole triangles.
    MalformedMesh {
        /// What was wrong with the buffers.
        reason: &'static str,
    },
    /// A triangle names a vertex the mesh does not have.
    VertexOutOfRange {
        /// The triangle that holds the index.
        triangle: usize,
        /// The index it holds.
        vertex: u64,
        /// How many vertices the mesh has.
        vertex_count: usize,
    },
    /// The surface or the output holds more elements than a 32-bit index.
    TooManyElements {
        /// What there were too many of.
        what: &'static str,
        /// How many were asked for.
        count: u64,
    },
}

impl fmt::Display for EmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositiveFinite { quantity, value } => {
                write!(f, "{quantity} must be finite and above zero, got {value}")
            }
            Self::NotFiniteNonNegative { quantity, value } => {
                write!(f, "{quantity} must be finite and not negative, got {value}")
            }
            Self::MalformedMesh { reason } => write!(f, "malformed mesh: {reason}"),
            Self::VertexOutOfRange {
                triangle,
                vertex,
                vertex_count,
            } => write!(
                f,
                "triangle {triangle} names vertex {vertex}, but the mesh has {vertex_count}"
            ),
            Self::TooManyElements { what, count } => {
                write!(f, "{count} {what} do not fit a 32-bit index")
            }
        }
    }
}

impl std::error::Error for EmitError {}
