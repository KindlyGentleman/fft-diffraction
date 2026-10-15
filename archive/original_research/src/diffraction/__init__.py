"""Europhotonics Numerical Diffraction Research Package.

Numerical Accuracy and Sampling Limitations of FFT-Based Scalar Diffraction Models.
"""

from diffraction.grids import Grid, create_grid
from diffraction.apertures import (
    make_circular_aperture,
    make_rectangular_aperture,
    make_double_slit_aperture,
)
from diffraction.propagation import (
    propagate_asm,
    propagate_blas,
    propagate_fresnel,
    propagate_direct_integration,
)
from diffraction.bandlimit import (
    compute_bandlimit_frequencies,
    make_bandlimit_mask,
    analyze_spectral_support,
)
from diffraction.analytical import (
    circular_fraunhofer_field,
    circular_fraunhofer_intensity,
    rectangular_fraunhofer_field,
    rectangular_fraunhofer_intensity,
    airy_first_zero_radius,
)
from diffraction.metrics import (
    calculate_intensity,
    calculate_energy,
    calculate_metrics,
    relative_l2_error,
)

__all__ = [
    "Grid",
    "create_grid",
    "make_circular_aperture",
    "make_rectangular_aperture",
    "make_double_slit_aperture",
    "propagate_asm",
    "propagate_blas",
    "propagate_fresnel",
    "propagate_direct_integration",
    "compute_bandlimit_frequencies",
    "make_bandlimit_mask",
    "analyze_spectral_support",
    "circular_fraunhofer_field",
    "circular_fraunhofer_intensity",
    "rectangular_fraunhofer_field",
    "rectangular_fraunhofer_intensity",
    "airy_first_zero_radius",
    "calculate_intensity",
    "calculate_energy",
    "calculate_metrics",
    "relative_l2_error",
]
