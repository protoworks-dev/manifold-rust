//! The read-only surface every emitter works from.

use manifold_rust::linalg::{cross, length, Vec3};
use manifold_rust::manifold::Manifold;
use manifold_rust::types::MeshGL64;

use crate::error::EmitError;
use crate::units::Area;

/// Coordinates per vertex position.
const POSITION_WIDTH: usize = 3;
/// Corners per triangle.
const TRIANGLE_CORNERS: usize = 3;
/// The kernel's "export no normals" argument to its mesh export.
const NO_NORMAL_CHANNEL: i32 = -1;
/// A triangle's area is this fraction of the length of its edge cross product.
const AREA_OF_CROSS: f64 = 0.5;

/// A vertex of a [`Surface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(u32);

impl VertexId {
    /// The vertex's position in the surface's vertex list.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A triangle of a [`Surface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TriangleId(u32);

impl TriangleId {
    /// The triangle's position in the surface's triangle list.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The same id as it is stored in a GPU buffer.
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// One triangle's corners with the frame an emitter orients its output by.
///
/// A triangle with no area has a zero `normal` and a zero `tangent`; emitters
/// give it no samples and no splat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleFrame {
    /// The corners, counter-clockwise seen from outside the solid.
    pub corners: [Vec3; 3],
    /// Unit normal, pointing out of the solid.
    pub normal: Vec3,
    /// Unit vector along the first edge, in the triangle's plane.
    pub tangent: Vec3,
    /// The triangle's area.
    pub area: f64,
}

impl TriangleFrame {
    fn new(corners: [Vec3; 3]) -> Self {
        let [a, b, c] = corners;
        let edge = b - a;
        let scaled_normal = cross(edge, c - a);
        let cross_length = length(scaled_normal);
        let edge_length = length(edge);
        if cross_length > 0.0 && edge_length > 0.0 {
            Self {
                corners,
                normal: scaled_normal / cross_length,
                tangent: edge / edge_length,
                area: AREA_OF_CROSS * cross_length,
            }
        } else {
            Self {
                corners,
                normal: Vec3::default(),
                tangent: Vec3::default(),
                area: 0.0,
            }
        }
    }

    /// The mean of the three corners.
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        let [a, b, c] = self.corners;
        (a + b + c) / TRIANGLE_CORNERS as f64
    }

    /// The second in-plane unit vector: `normal` crossed with `tangent`.
    #[must_use]
    pub fn bitangent(&self) -> Vec3 {
        cross(self.normal, self.tangent)
    }

    /// The point at the weights `u` on the second corner and `v` on the third.
    #[must_use]
    pub fn point(&self, u: f64, v: f64) -> Vec3 {
        let [a, b, c] = self.corners;
        a + (b - a) * u + (c - a) * v
    }
}

/// An evaluated solid's boundary: positions, triangles and their areas.
///
/// A surface is built once and then only read, so any number of emitters can
/// share it.
#[derive(Clone, Debug, Default)]
pub struct Surface {
    positions: Vec<Vec3>,
    triangles: Vec<[VertexId; 3]>,
    /// Sum of the areas of triangles `0..=i`.
    cumulative_area: Vec<f64>,
}

impl Surface {
    /// The boundary of an evaluated solid.
    ///
    /// # Errors
    ///
    /// Refuses a solid with more vertices or triangles than a 32-bit index.
    pub fn from_manifold(solid: &Manifold) -> Result<Self, EmitError> {
        Self::from_mesh(&solid.get_mesh_gl64(NO_NORMAL_CHANNEL))
    }

    /// The surface a kernel mesh describes. Only positions and triangles are
    /// read; further vertex properties are left to the caller.
    ///
    /// # Errors
    ///
    /// Refuses buffers that do not hold whole vertices or whole triangles, a
    /// triangle that names a missing vertex, and more vertices or triangles
    /// than a 32-bit index.
    pub fn from_mesh(mesh: &MeshGL64) -> Result<Self, EmitError> {
        let width = usize::try_from(mesh.num_prop).map_err(|_| EmitError::MalformedMesh {
            reason: "property count does not fit this platform",
        })?;
        if mesh.vert_properties.is_empty() && mesh.tri_verts.is_empty() {
            return Ok(Self::default());
        }
        if width < POSITION_WIDTH {
            return Err(EmitError::MalformedMesh {
                reason: "fewer than three properties per vertex",
            });
        }
        if mesh.vert_properties.len() % width != 0 {
            return Err(EmitError::MalformedMesh {
                reason: "vertex properties are not a whole number of vertices",
            });
        }
        if mesh.tri_verts.len() % TRIANGLE_CORNERS != 0 {
            return Err(EmitError::MalformedMesh {
                reason: "triangle indices are not a whole number of triangles",
            });
        }
        let vertex_count = mesh.vert_properties.len() / width;
        let triangle_count = mesh.tri_verts.len() / TRIANGLE_CORNERS;
        check_count("vertices", vertex_count)?;
        check_count("triangles", triangle_count)?;

        let positions: Vec<Vec3> = mesh
            .vert_properties
            .chunks_exact(width)
            .map(|vertex| Vec3::new(vertex[0], vertex[1], vertex[2]))
            .collect();

        let mut triangles = Vec::with_capacity(triangle_count);
        for (triangle, corners) in mesh.tri_verts.chunks_exact(TRIANGLE_CORNERS).enumerate() {
            let mut ids = [VertexId(0); TRIANGLE_CORNERS];
            for (id, &vertex) in ids.iter_mut().zip(corners) {
                let in_range = u32::try_from(vertex)
                    .ok()
                    .filter(|&v| (v as usize) < vertex_count);
                let Some(v) = in_range else {
                    return Err(EmitError::VertexOutOfRange {
                        triangle,
                        vertex,
                        vertex_count,
                    });
                };
                *id = VertexId(v);
            }
            triangles.push(ids);
        }

        let mut surface = Self {
            positions,
            triangles,
            cumulative_area: Vec::with_capacity(triangle_count),
        };
        let mut running = 0.0;
        for index in 0..triangle_count {
            running += surface.frame_at(index).area;
            surface.cumulative_area.push(running);
        }
        Ok(surface)
    }

    /// How many vertices the surface has.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// How many triangles the surface has.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Every triangle id, in order.
    pub fn triangle_ids(&self) -> impl Iterator<Item = TriangleId> + '_ {
        // The count was checked against `u32::MAX` when the surface was built.
        (0..self.triangles.len()).map(|index| TriangleId(index as u32))
    }

    /// The vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// The triangles as vertex ids, counter-clockwise seen from outside.
    #[must_use]
    pub fn triangles(&self) -> &[[VertexId; 3]] {
        &self.triangles
    }

    /// The corners and frame of one triangle.
    #[must_use]
    pub fn frame(&self, triangle: TriangleId) -> TriangleFrame {
        self.frame_at(triangle.index())
    }

    /// The whole surface's area.
    #[must_use]
    pub fn area(&self) -> Area {
        // A sum of non-negative finite areas; `Area::new` cannot refuse it
        // unless the mesh held a non-finite coordinate, which reads as zero.
        let total = self.cumulative_area.last().copied().unwrap_or(0.0);
        Area::new(total).unwrap_or_default()
    }

    /// The summed area of triangles `0..=i`, for each triangle `i`.
    pub(crate) fn cumulative_area(&self) -> &[f64] {
        &self.cumulative_area
    }

    /// The id of the triangle at `index`, which the caller has from this
    /// surface's own triangle range.
    pub(crate) fn id_at(index: usize) -> TriangleId {
        TriangleId(index as u32)
    }

    fn frame_at(&self, index: usize) -> TriangleFrame {
        let [a, b, c] = self.triangles[index];
        TriangleFrame::new([
            self.positions[a.index()],
            self.positions[b.index()],
            self.positions[c.index()],
        ])
    }
}

fn check_count(what: &'static str, count: usize) -> Result<(), EmitError> {
    if u32::try_from(count).is_ok() {
        Ok(())
    } else {
        Err(EmitError::TooManyElements {
            what,
            count: count as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube() -> Surface {
        let cube = Manifold::cube(Vec3::new(1.0, 1.0, 1.0), true);
        Surface::from_manifold(&cube).expect("a cube is a valid surface")
    }

    #[test]
    fn cube_has_twelve_triangles_and_area_six() {
        let surface = unit_cube();
        assert_eq!(surface.triangle_count(), 12);
        assert_eq!(surface.area().get().to_bits(), 6.0_f64.to_bits());
    }

    #[test]
    fn cube_normals_point_outward() {
        let surface = unit_cube();
        for id in surface.triangle_ids() {
            let frame = surface.frame(id);
            let outward = manifold_rust::linalg::dot(frame.normal, frame.centroid());
            assert!(outward > 0.0, "triangle {} faces inward", id.index());
        }
    }

    #[test]
    fn empty_solid_is_an_empty_surface() {
        let surface = Surface::from_manifold(&Manifold::empty()).expect("empty is valid");
        assert_eq!(surface.triangle_count(), 0);
        assert_eq!(surface.area().get().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn out_of_range_vertex_is_refused() {
        let mesh = MeshGL64 {
            num_prop: 3,
            vert_properties: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            tri_verts: vec![0, 1, 3],
            ..MeshGL64::default()
        };
        assert_eq!(
            Surface::from_mesh(&mesh).err(),
            Some(EmitError::VertexOutOfRange {
                triangle: 0,
                vertex: 3,
                vertex_count: 3,
            })
        );
    }
}
