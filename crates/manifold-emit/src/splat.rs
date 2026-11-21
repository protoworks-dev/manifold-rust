//! Surface-aligned Gaussian splats, computed from the triangles.
//!
//! Each splat is a flat Gaussian lying in its triangle's plane: two in-plane
//! standard deviations and a third, the thickness, along the normal. Nothing
//! is trained; the splats are a closed-form function of the surface. Colour
//! and opacity belong to the material the renderer assigns, so this output
//! carries geometry and the source triangle only.

use manifold_rust::linalg::{cross, dot, Vec3};

use crate::error::EmitError;
use crate::par::map_indexed;
use crate::sampling::{narrow, Allocation};
use crate::surface::{Surface, TriangleFrame, TriangleId};
use crate::units::{Length, SurfaceDensity};
use crate::Emit;

/// A uniform triangle's covariance is the sum over its corners of the outer
/// product of each corner's offset from the centroid, divided by this.
const TRIANGLE_COVARIANCE_DIVISOR: f64 = 12.0;

/// A sampled splat's in-plane standard deviation, in sample spacings.
///
/// At `1.0` a splat's standard deviation equals the distance between
/// neighbouring samples. Smaller values leave gaps; larger values blur.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SplatOverlap(f64);

impl SplatOverlap {
    /// An overlap of `value` sample spacings.
    ///
    /// # Errors
    ///
    /// Refuses a value that is not finite or not above zero.
    pub fn new(value: f64) -> Result<Self, EmitError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(EmitError::NotPositiveFinite {
                quantity: "splat overlap",
                value,
            })
        }
    }

    /// The overlap in sample spacings.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// Where splats are placed on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplatLayout {
    /// One splat per triangle, fitted to the triangle's own centroid and
    /// covariance. The count follows the tessellation, so a large flat face
    /// gets few, wide splats.
    PerTriangle,
    /// Round splats at evenly spread samples, independent of tessellation.
    Sampled {
        /// Splats per unit of area.
        density: SurfaceDensity,
        /// Each splat's in-plane size against the sample spacing.
        overlap: SplatOverlap,
    },
}

/// Places flat Gaussian splats on a surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplatEmitter {
    layout: SplatLayout,
    thickness: Length,
}

impl SplatEmitter {
    /// An emitter that lays splats out by `layout`, each with the standard
    /// deviation `thickness` along its normal.
    #[must_use]
    pub fn new(layout: SplatLayout, thickness: Length) -> Self {
        Self { layout, thickness }
    }
}

/// Splats on a surface, as parallel columns ready for GPU buffers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplatCloud {
    positions: Vec<[f32; 3]>,
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
    triangles: Vec<TriangleId>,
}

impl SplatCloud {
    /// How many splats the cloud holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the cloud holds no splats.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Each splat's centre.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// Each splat's orientation as a unit quaternion `[x, y, z, w]` with
    /// `w >= 0`. It turns the local z axis onto the surface normal and the
    /// local x axis onto the splat's longer in-plane axis.
    #[must_use]
    pub fn rotations(&self) -> &[[f32; 4]] {
        &self.rotations
    }

    /// Each splat's standard deviations along its local x, y and z axes, in
    /// model units: the longer in-plane axis, the shorter, then the thickness.
    #[must_use]
    pub fn scales(&self) -> &[[f32; 3]] {
        &self.scales
    }

    /// The surface triangle each splat lies on.
    #[must_use]
    pub fn triangles(&self) -> &[TriangleId] {
        &self.triangles
    }
}

/// One splat before it is split into columns.
struct Splat {
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    triangle: TriangleId,
}

impl Splat {
    /// A splat at `centre` whose local x axis is `major`, in the plane of
    /// `frame`.
    fn new(
        triangle: TriangleId,
        frame: &TriangleFrame,
        centre: Vec3,
        major: Vec3,
        deviations: [f64; 2],
        thickness: Length,
    ) -> Self {
        let minor = cross(frame.normal, major);
        Self {
            position: narrow(centre),
            rotation: quaternion([major, minor, frame.normal]),
            scale: [
                deviations[0] as f32,
                deviations[1] as f32,
                thickness.get() as f32,
            ],
            triangle,
        }
    }

    /// The splat with the centroid and covariance of a whole triangle.
    fn fitted(triangle: TriangleId, frame: &TriangleFrame, thickness: Length) -> Self {
        let centre = frame.centroid();
        let bitangent = frame.bitangent();
        let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
        for corner in frame.corners {
            let offset = corner - centre;
            let x = dot(offset, frame.tangent);
            let y = dot(offset, bitangent);
            xx += x * x;
            xy += x * y;
            yy += y * y;
        }
        let (xx, xy, yy) = (
            xx / TRIANGLE_COVARIANCE_DIVISOR,
            xy / TRIANGLE_COVARIANCE_DIVISOR,
            yy / TRIANGLE_COVARIANCE_DIVISOR,
        );
        // Eigenvalues and the major eigenvector of the two by two covariance.
        let mean = (xx + yy) / 2.0;
        let half_difference = (xx - yy) / 2.0;
        let radius = half_difference.hypot(xy);
        let angle = xy.atan2(half_difference) / 2.0;
        let major = frame.tangent * angle.cos() + bitangent * angle.sin();
        let deviations = [(mean + radius).sqrt(), (mean - radius).max(0.0).sqrt()];
        Self::new(triangle, frame, centre, major, deviations, thickness)
    }
}

/// The unit quaternion `[x, y, z, w]`, `w >= 0`, of the rotation whose matrix
/// has the three orthonormal `axes` as its columns (Shepperd's method).
fn quaternion(axes: [Vec3; 3]) -> [f32; 4] {
    let [x, y, z] = axes;
    let trace = x.x + y.y + z.z;
    let [qx, qy, qz, qw] = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [(y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s / 4.0]
    } else if x.x > y.y && x.x > z.z {
        let s = (1.0 + x.x - y.y - z.z).sqrt() * 2.0;
        [s / 4.0, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s]
    } else if y.y > z.z {
        let s = (1.0 + y.y - x.x - z.z).sqrt() * 2.0;
        [(y.x + x.y) / s, s / 4.0, (z.y + y.z) / s, (z.x - x.z) / s]
    } else {
        let s = (1.0 + z.z - x.x - y.y).sqrt() * 2.0;
        [(z.x + x.z) / s, (z.y + y.z) / s, s / 4.0, (x.y - y.x) / s]
    };
    let sign = if qw < 0.0 { -1.0 } else { 1.0 };
    [
        (qx * sign) as f32,
        (qy * sign) as f32,
        (qz * sign) as f32,
        (qw * sign) as f32,
    ]
}

impl Emit for SplatEmitter {
    type Output = SplatCloud;

    fn emit(&self, surface: &Surface) -> Result<SplatCloud, EmitError> {
        let thickness = self.thickness;
        let splats = match self.layout {
            SplatLayout::PerTriangle => {
                let solid: Vec<TriangleId> = surface
                    .triangle_ids()
                    .filter(|&id| surface.frame(id).area > 0.0)
                    .collect();
                map_indexed(solid.len(), |k| {
                    let triangle = solid[k];
                    Splat::fitted(triangle, &surface.frame(triangle), thickness)
                })
            }
            SplatLayout::Sampled { density, overlap } => {
                let allocation = Allocation::new(surface, density)?;
                let deviation = overlap.get() * density.spacing();
                map_indexed(allocation.len(), |k| {
                    let site = allocation.site(k);
                    Splat::new(
                        site.triangle,
                        &site.frame,
                        site.position,
                        site.frame.tangent,
                        [deviation, deviation],
                        thickness,
                    )
                })
            }
        };
        let mut cloud = SplatCloud {
            positions: Vec::with_capacity(splats.len()),
            rotations: Vec::with_capacity(splats.len()),
            scales: Vec::with_capacity(splats.len()),
            triangles: Vec::with_capacity(splats.len()),
        };
        for splat in splats {
            cloud.positions.push(splat.position);
            cloud.rotations.push(splat.rotation);
            cloud.scales.push(splat.scale);
            cloud.triangles.push(splat.triangle);
        }
        Ok(cloud)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_rust::manifold::Manifold;

    /// The thickness the tests give every splat.
    const THICKNESS: f64 = 0.001;
    /// How far a quaternion's length may sit from one after `f32` narrowing.
    const UNIT_TOLERANCE: f32 = 1.0e-5;

    fn cube_surface() -> Surface {
        let cube = Manifold::cube(Vec3::new(1.0, 1.0, 1.0), true);
        Surface::from_manifold(&cube).expect("a cube is a valid surface")
    }

    fn emit(layout: SplatLayout) -> SplatCloud {
        let thickness = Length::new(THICKNESS).expect("a positive thickness");
        SplatEmitter::new(layout, thickness)
            .emit(&cube_surface())
            .expect("the cloud fits")
    }

    /// The splat's local axis `axis`, turned by its quaternion.
    fn rotated(q: [f32; 4], axis: [f32; 3]) -> [f32; 3] {
        let [x, y, z, w] = q;
        let u = [x, y, z];
        let c1 = [
            u[1] * axis[2] - u[2] * axis[1],
            u[2] * axis[0] - u[0] * axis[2],
            u[0] * axis[1] - u[1] * axis[0],
        ];
        let c2 = [
            u[1] * c1[2] - u[2] * c1[1],
            u[2] * c1[0] - u[0] * c1[2],
            u[0] * c1[1] - u[1] * c1[0],
        ];
        [0, 1, 2].map(|i| axis[i] + 2.0 * (w * c1[i] + c2[i]))
    }

    #[test]
    fn per_triangle_gives_one_splat_each() {
        let cloud = emit(SplatLayout::PerTriangle);
        assert_eq!(cloud.len(), 12);
        for scale in cloud.scales() {
            assert!(scale[0] >= scale[1] && scale[1] > 0.0);
            assert_eq!(scale[2].to_bits(), (THICKNESS as f32).to_bits());
        }
    }

    #[test]
    fn rotations_are_unit_and_carry_z_onto_the_normal() {
        let surface = cube_surface();
        let cloud = emit(SplatLayout::PerTriangle);
        for (rotation, triangle) in cloud.rotations().iter().zip(cloud.triangles()) {
            let length = rotation.iter().map(|c| c * c).sum::<f32>().sqrt();
            assert!((length - 1.0).abs() < UNIT_TOLERANCE);
            assert!(rotation[3] >= 0.0);
            let normal = narrow(surface.frame(*triangle).normal);
            let z = rotated(*rotation, [0.0, 0.0, 1.0]);
            for (got, want) in z.iter().zip(normal) {
                assert!((got - want).abs() < UNIT_TOLERANCE, "{z:?} vs {normal:?}");
            }
        }
    }

    #[test]
    fn right_triangle_covariance_matches_the_closed_form() {
        // Half of a unit square: legs of length one along the two in-plane
        // axes. Its covariance has eigenvalues 1/12 and 1/36.
        let cloud = emit(SplatLayout::PerTriangle);
        let major = (1.0_f64 / 12.0).sqrt() as f32;
        let minor = (1.0_f64 / 36.0).sqrt() as f32;
        for scale in cloud.scales() {
            assert!((scale[0] - major).abs() < UNIT_TOLERANCE);
            assert!((scale[1] - minor).abs() < UNIT_TOLERANCE);
        }
    }

    #[test]
    fn sampled_layout_follows_the_density() {
        let density = SurfaceDensity::new(100.0).expect("a positive density");
        let overlap = SplatOverlap::new(1.0).expect("a positive overlap");
        let cloud = emit(SplatLayout::Sampled { density, overlap });
        assert_eq!(cloud.len(), 600);
        // 100 per unit area is a spacing of one tenth.
        let spacing = 0.1_f32;
        for scale in cloud.scales() {
            assert_eq!(scale[0].to_bits(), spacing.to_bits());
            assert_eq!(scale[1].to_bits(), spacing.to_bits());
        }
    }

    #[test]
    fn emission_is_repeatable() {
        assert_eq!(
            emit(SplatLayout::PerTriangle),
            emit(SplatLayout::PerTriangle)
        );
    }
}
