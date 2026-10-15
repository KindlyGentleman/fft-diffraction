from typing import Tuple
import numpy as np
from diffraction.grids import Grid

def make_circular_aperture(
    grid: Grid,
    radius: float,
    center: Tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a binary circular aperture transmission field on the given grid."""
    if radius <= 0.0:
        raise ValueError(f"Aperture radius must be positive, got {radius}")

    x0, y0 = center
    r_sq = (grid.X - x0) ** 2 + (grid.Y - y0) ** 2
    field = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    mask = r_sq <= (radius**2)
    field[mask] = complex(amplitude, 0.0)
    return field

def make_rectangular_aperture(
    grid: Grid,
    width: float,
    height: float,
    center: Tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a binary rectangular aperture transmission field on the given grid."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"Width and height must be positive, got width={width}, height={height}")

    x0, y0 = center
    field = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    mask = (np.abs(grid.X - x0) <= (width / 2.0)) & (np.abs(grid.Y - y0) <= (height / 2.0))
    field[mask] = complex(amplitude, 0.0)
    return field

def make_double_slit_aperture(
    grid: Grid,
    slit_width: float,
    slit_separation: float,
    slit_height: float,
    center: Tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a double slit aperture transmission field on the given grid."""
    if slit_width <= 0.0 or slit_separation <= 0.0 or slit_height <= 0.0:
        raise ValueError("Dimensions must be positive")

    x0, y0 = center
    field = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    left_center = x0 - slit_separation / 2.0
    right_center = x0 + slit_separation / 2.0

    mask_left = (np.abs(grid.X - left_center) <= (slit_width / 2.0)) & (
        np.abs(grid.Y - y0) <= (slit_height / 2.0)
    )
    mask_right = (np.abs(grid.X - right_center) <= (slit_width / 2.0)) & (
        np.abs(grid.Y - y0) <= (slit_height / 2.0)
    )
    field[mask_left | mask_right] = complex(amplitude, 0.0)
    return field
