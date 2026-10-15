from typing import Tuple
import numpy as np
from scipy.special import j1
from diffraction.grids import Grid

FIRST_AIRY_ZERO_Q = 3.831705970207512

def jinc(q: np.ndarray) -> np.ndarray:
    """Evaluate 2 * J1(q) / q with accurate limit as q -> 0."""
    out = np.ones_like(q, dtype=np.float64)
    nonzero = np.abs(q) > 1e-12
    q_nz = q[nonzero]
    out[nonzero] = 2.0 * j1(q_nz) / q_nz
    return out

def airy_first_zero_radius(wavelength: float, z: float, radius: float) -> float:
    """Theoretical radius of the first dark ring in meters."""
    k = 2.0 * np.pi / wavelength
    return float(FIRST_AIRY_ZERO_Q * z / (k * radius))

def circular_fraunhofer_field(
    grid: Grid,
    wavelength: float,
    z: float,
    radius: float,
    incident_amplitude: float = 1.0,
) -> np.ndarray:
    """Exact complex optical field for Fraunhofer circular aperture diffraction."""
    if z <= 0.0 or radius <= 0.0 or wavelength <= 0.0:
        raise ValueError("z, radius, and wavelength must be positive")

    k = 2.0 * np.pi / wavelength
    r = np.sqrt(grid.X**2 + grid.Y**2)
    q = k * radius * r / z

    aperture_area = np.pi * (radius**2)
    factor = (
        np.exp(1j * k * z)
        * np.exp(1j * k / (2.0 * z) * (r**2))
        / (1j * wavelength * z)
    )
    return factor * incident_amplitude * aperture_area * jinc(q)

def circular_fraunhofer_intensity(
    grid: Grid,
    wavelength: float,
    z: float,
    radius: float,
    incident_amplitude: float = 1.0,
) -> np.ndarray:
    """Intensity distribution for Fraunhofer circular aperture diffraction."""
    field = circular_fraunhofer_field(grid, wavelength, z, radius, incident_amplitude)
    return np.abs(field) ** 2

def rectangular_fraunhofer_field(
    grid: Grid,
    wavelength: float,
    z: float,
    width: float,
    height: float,
    incident_amplitude: float = 1.0,
) -> np.ndarray:
    """Exact complex optical field for Fraunhofer rectangular aperture diffraction."""
    if z <= 0.0 or width <= 0.0 or height <= 0.0 or wavelength <= 0.0:
        raise ValueError("z, width, height, and wavelength must be positive")

    k = 2.0 * np.pi / wavelength
    r_sq = grid.X**2 + grid.Y**2

    factor = (
        np.exp(1j * k * z)
        * np.exp(1j * k / (2.0 * z) * r_sq)
        / (1j * wavelength * z)
    )
    sinc_x = np.sinc(width * grid.X / (wavelength * z))
    sinc_y = np.sinc(height * grid.Y / (wavelength * z))
    return factor * incident_amplitude * (width * height) * sinc_x * sinc_y

def rectangular_fraunhofer_intensity(
    grid: Grid,
    wavelength: float,
    z: float,
    width: float,
    height: float,
    incident_amplitude: float = 1.0,
) -> np.ndarray:
    """Intensity distribution for Fraunhofer rectangular aperture diffraction."""
    field = rectangular_fraunhofer_field(grid, wavelength, z, width, height, incident_amplitude)
    return np.abs(field) ** 2
