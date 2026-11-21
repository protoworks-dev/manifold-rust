//! A point cloud sampled from a surface.

use crate::error::EmitError;
use crate::par::map_indexed;
use crate::sampling::{narrow, Allocation};
use crate::surface::{Surface, TriangleId};
use crate::units::SurfaceDensity;
use crate::Emit;

/// Samples a surface at a fixed number of points per unit area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCloudEmitter {
    density: SurfaceDensity,
}

impl PointCloudEmitter {
    /// An emitter that places `density` points on each unit of area.
    #[must_use]
    pub fn new(density: SurfaceDensity) -> Self {
        Self { density }
    }
}

/// Points on a surface, as parallel columns ready for GPU buffers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PointCloud {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    triangles: Vec<TriangleId>,
}

impl PointCloud {
    /// How many points the cloud holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the cloud holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Each point's position.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// Each point's unit normal, pointing out of the solid.
    #[must_use]
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }

    /// The surface triangle each point lies on.
    #[must_use]
    pub fn triangles(&self) -> &[TriangleId] {
        &self.triangles
    }
}

impl Emit for PointCloudEmitter {
    type Output = PointCloud;

    fn emit(&self, surface: &Surface) -> Result<PointCloud, EmitError> {
        let allocation = Allocation::new(surface, self.density)?;
        let sites = map_indexed(allocation.len(), |k| {
            let site = allocation.site(k);
            (
                narrow(site.position),
                narrow(site.frame.normal),
                site.triangle,
            )
        });
        let mut cloud = PointCloud {
            positions: Vec::with_capacity(sites.len()),
            normals: Vec::with_capacity(sites.len()),
            triangles: Vec::with_capacity(sites.len()),
        };
        for (position, normal, triangle) in sites {
            cloud.positions.push(position);
            cloud.normals.push(normal);
            cloud.triangles.push(triangle);
        }
        Ok(cloud)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_rust::linalg::Vec3;
    use manifold_rust::manifold::Manifold;

    /// Half the side of the centred unit cube the tests sample.
    const HALF_SIDE: f32 = 0.5;

    fn cube_cloud(density: f64) -> PointCloud {
        let cube = Manifold::cube(Vec3::new(1.0, 1.0, 1.0), true);
        let surface = Surface::from_manifold(&cube).expect("a cube is a valid surface");
        let density = SurfaceDensity::new(density).expect("a positive density");
        PointCloudEmitter::new(density)
            .emit(&surface)
            .expect("the cloud fits")
    }

    #[test]
    fn count_is_area_times_density() {
        assert_eq!(cube_cloud(100.0).len(), 600);
        assert_eq!(cube_cloud(0.5).len(), 3);
    }

    #[test]
    fn every_point_lies_on_a_cube_face() {
        let cloud = cube_cloud(100.0);
        for (position, normal) in cloud.positions().iter().zip(cloud.normals()) {
            // A face of the centred cube holds one coordinate at exactly half
            // a side, on the axis its normal points along.
            let on_face = position
                .iter()
                .zip(normal)
                .any(|(p, n)| p.abs().to_bits() == HALF_SIDE.to_bits() && n.abs() > HALF_SIDE);
            assert!(on_face, "{position:?} is off the cube");
            assert!(position.iter().all(|p| p.abs() <= HALF_SIDE));
        }
    }

    #[test]
    fn each_face_receives_its_share() {
        let cloud = cube_cloud(100.0);
        // Six faces of unit area at 100 points each; a face is two triangles,
        // and rounding moves at most one point across a triangle boundary.
        for axis in 0..3 {
            for sign in [-1.0_f32, 1.0] {
                let count = cloud
                    .normals()
                    .iter()
                    .filter(|n| n[axis] * sign > HALF_SIDE)
                    .count();
                assert!((99..=101).contains(&count), "face holds {count}");
            }
        }
    }

    #[test]
    fn emission_is_repeatable() {
        assert_eq!(cube_cloud(250.0), cube_cloud(250.0));
    }
}
