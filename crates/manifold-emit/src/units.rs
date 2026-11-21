//! Checked quantities, in the model's own length unit.
//!
//! The kernel carries no unit; a model built in metres gives lengths in
//! metres, areas in square metres and densities in points per square metre.

use crate::error::EmitError;

fn positive_finite(quantity: &'static str, value: f64) -> Result<f64, EmitError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EmitError::NotPositiveFinite { quantity, value })
    }
}

/// A length greater than zero.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Length(f64);

impl Length {
    /// A length of `value` model units.
    ///
    /// # Errors
    ///
    /// Refuses a value that is not finite or not above zero.
    pub fn new(value: f64) -> Result<Self, EmitError> {
        positive_finite("length", value).map(Self)
    }

    /// The length in model units.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// An area of zero or more; the default is zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Area(f64);

impl Area {
    /// An area of `value` square model units.
    ///
    /// # Errors
    ///
    /// Refuses a value that is not finite or is negative.
    pub fn new(value: f64) -> Result<Self, EmitError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(EmitError::NotFiniteNonNegative {
                quantity: "area",
                value,
            })
        }
    }

    /// The area in square model units.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }
}

/// How many samples one unit of area receives.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SurfaceDensity(f64);

impl SurfaceDensity {
    /// A density of `value` samples per square model unit.
    ///
    /// # Errors
    ///
    /// Refuses a value that is not finite or not above zero.
    pub fn new(value: f64) -> Result<Self, EmitError> {
        positive_finite("surface density", value).map(Self)
    }

    /// Samples per square model unit.
    #[must_use]
    pub fn get(self) -> f64 {
        self.0
    }

    /// The side of the square that holds one sample at this density.
    #[must_use]
    pub fn spacing(self) -> f64 {
        self.0.sqrt().recip()
    }
}
