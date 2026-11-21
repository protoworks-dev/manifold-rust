//! Deterministic surface sampling shared by the emitters.
//!
//! Samples are laid along the triangle list by cumulative area, so the count
//! in any run of triangles is the run's area times the density, rounded, and
//! the total never drifts from the surface's area. Inside its triangle a
//! sample takes the next point of a low-discrepancy sequence, indexed by the
//! sample's own number, so sample `k` depends on `k` alone and the whole set
//! can be computed in any order.

use manifold_rust::linalg::Vec3;

use crate::error::EmitError;
use crate::surface::{Surface, TriangleFrame, TriangleId};
use crate::units::SurfaceDensity;

/// Two to the power 64, the range the sequence's fixed-point steps wrap in.
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
/// The plastic number's reciprocal and its square: the steps of the R2
/// sequence (Roberts 2018, "The Unreasonable Effectiveness of Quasirandom
/// Sequences"), the lowest-discrepancy additive recurrence in two dimensions.
const R2_STEP: [f64; 2] = [0.754_877_666_246_692_7, 0.569_840_290_998_053_2];
/// The same steps as 64-bit fixed-point fractions. Multiplying by a sample
/// number wraps modulo one exactly, on every platform.
const R2_STEP_FIXED: [u64; 2] = [
    (R2_STEP[0] * TWO_POW_64) as u64,
    (R2_STEP[1] * TWO_POW_64) as u64,
];
/// Bits a `u64` fraction drops to fit an `f64` mantissa.
const MANTISSA_SHIFT: u32 = 11;
/// One unit in the last place of a 53-bit fraction.
const MANTISSA_UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Point `k` of the R2 sequence in the unit square.
fn unit_square(k: u64) -> [f64; 2] {
    let n = k.wrapping_add(1);
    R2_STEP_FIXED.map(|step| (n.wrapping_mul(step) >> MANTISSA_SHIFT) as f64 * MANTISSA_UNIT)
}

/// Point `k` of the sequence, folded from the unit square onto the weights of
/// a triangle's second and third corner.
fn triangle_weights(k: u64) -> [f64; 2] {
    let [u, v] = unit_square(k);
    if u + v > 1.0 {
        [1.0 - u, 1.0 - v]
    } else {
        [u, v]
    }
}

/// One sample: where it is and the triangle it lies on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Site {
    pub(crate) triangle: TriangleId,
    pub(crate) frame: TriangleFrame,
    pub(crate) position: Vec3,
}

/// Which triangle each sample of a surface falls on, at one density.
pub(crate) struct Allocation<'a> {
    surface: &'a Surface,
    /// One past the last sample number of triangles `0..=i`.
    ends: Vec<u64>,
}

impl<'a> Allocation<'a> {
    /// Allocate samples over `surface` at `density`.
    pub(crate) fn new(surface: &'a Surface, density: SurfaceDensity) -> Result<Self, EmitError> {
        let ends: Vec<u64> = surface
            .cumulative_area()
            .iter()
            // Areas are finite and not negative, so the product rounds to a
            // whole number of samples that never decreases along the list.
            .map(|area| (area * density.get()).round() as u64)
            .collect();
        let total = ends.last().copied().unwrap_or(0);
        if u32::try_from(total).is_err() {
            return Err(EmitError::TooManyElements {
                what: "samples",
                count: total,
            });
        }
        Ok(Self { surface, ends })
    }

    /// How many samples the surface receives.
    pub(crate) fn len(&self) -> usize {
        // Checked against `u32::MAX` in `new`.
        self.ends.last().copied().unwrap_or(0) as usize
    }

    /// Sample number `k`, for `k` below [`Self::len`].
    pub(crate) fn site(&self, k: usize) -> Site {
        let number = k as u64;
        let index = self.ends.partition_point(|&end| end <= number);
        let triangle = Surface::id_at(index);
        let frame = self.surface.frame(triangle);
        let [u, v] = triangle_weights(number);
        Site {
            triangle,
            frame,
            position: frame.point(u, v),
        }
    }
}

/// A kernel vector narrowed to the `f32` triple a GPU buffer holds.
pub(crate) fn narrow(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_stays_in_the_unit_square() {
        for k in 0..10_000 {
            let [u, v] = unit_square(k);
            assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn weights_stay_inside_the_triangle() {
        for k in 0..10_000 {
            let [u, v] = triangle_weights(k);
            assert!(u >= 0.0 && v >= 0.0 && u + v <= 1.0);
        }
    }

    #[test]
    fn sequence_fills_the_square_evenly() {
        // A four by four grid of cells over 1,600 points holds 100 each on
        // average; a low-discrepancy sequence keeps every cell near that.
        const CELLS: usize = 4;
        const POINTS: u64 = 1_600;
        let mut counts = [[0_u32; CELLS]; CELLS];
        for k in 0..POINTS {
            let [u, v] = unit_square(k);
            counts[(u * CELLS as f64) as usize][(v * CELLS as f64) as usize] += 1;
        }
        for row in counts {
            for count in row {
                assert!((90..=110).contains(&count), "cell holds {count}");
            }
        }
    }
}
