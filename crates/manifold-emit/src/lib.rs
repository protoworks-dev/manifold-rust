//! Output emitters over a `manifold-rust` solid.
//!
//! The kernel evaluates a model to one watertight triangle mesh. A renderer,
//! a point-based pipeline and a splat pipeline each want a different form of
//! that same surface. This crate gives them one read-only [`Surface`] and one
//! trait, [`Emit`], that every output implements:
//!
//! - [`PointCloudEmitter`] samples the surface at a stated density.
//! - [`SplatEmitter`] places surface-aligned Gaussian splats, computed from
//!   the triangles in closed form with no training step.
//!
//! An output that is not here implements [`Emit`] in its own crate.
//!
//! Every emitter is deterministic: the same surface and the same parameters
//! give the same bytes, with or without the `parallel` feature. Coordinates
//! are in the model's own length unit and stay in `f64` until the output
//! buffers, which are `f32` for the GPU.
//!
//! ```
//! use manifold_emit::{Emit, PointCloudEmitter, Surface, SurfaceDensity};
//! use manifold_rust::linalg::Vec3;
//! use manifold_rust::manifold::Manifold;
//!
//! let cube = Manifold::cube(Vec3::new(1.0, 1.0, 1.0), true);
//! let surface = Surface::from_manifold(&cube)?;
//! let cloud = PointCloudEmitter::new(SurfaceDensity::new(100.0)?).emit(&surface)?;
//! assert_eq!(cloud.len(), 600);
//! # Ok::<(), manifold_emit::EmitError>(())
//! ```

mod error;
mod par;
mod point_cloud;
mod sampling;
mod splat;
mod surface;
mod units;

pub use error::EmitError;
pub use point_cloud::{PointCloud, PointCloudEmitter};
pub use splat::{SplatCloud, SplatEmitter, SplatLayout, SplatOverlap};
pub use surface::{Surface, TriangleFrame, TriangleId, VertexId};
pub use units::{Area, Length, SurfaceDensity};

/// One output form of a surface.
///
/// An emitter holds its parameters and reads the surface; it never changes
/// it, so several emitters can run over one surface at the same time.
pub trait Emit {
    /// What this emitter produces.
    type Output;

    /// Produce the output for `surface`.
    ///
    /// # Errors
    ///
    /// Returns an [`EmitError`] when the output cannot be represented, for
    /// example when it would hold more elements than its index type allows.
    fn emit(&self, surface: &Surface) -> Result<Self::Output, EmitError>;
}
