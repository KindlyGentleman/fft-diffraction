from typing import Tuple, Dict
import numpy as np
from diffraction.grids import Grid

def compute_bandlimit_frequencies(
    wavelength: float,
    z: float,
    Sx: float,
    Sy: float,
) -> Tuple[float, float]:
    """Compute the Matsushima-Shimobaba critical cutoff frequencies.
    
    References:
        K. Matsushima and T. Shimobaba, Opt. Express 17(22), 19662-19673 (2009), Eq. (13), Eq. (20).
    """
    if wavelength <= 0.0:
        raise ValueError(f"Wavelength must be positive, got {wavelength}")
    if z < 0.0:
        raise ValueError(f"Propagation distance must be non-negative, got {z}")

    if z == 0.0:
        return 1.0 / wavelength, 1.0 / wavelength

    fx_limit = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * z / Sx) ** 2))
    fy_limit = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * z / Sy) ** 2))
    return fx_limit, fy_limit

def make_bandlimit_mask(
    grid: Grid,
    wavelength: float,
    z: float,
) -> np.ndarray:
    """Generate the binary 2D spatial frequency band-limiting mask."""
    fx_limit, fy_limit = compute_bandlimit_frequencies(
        wavelength=wavelength,
        z=z,
        Sx=grid.Sx,
        Sy=grid.Sy,
    )
    mask = (np.abs(grid.FX) <= fx_limit) & (np.abs(grid.FY) <= fy_limit)
    return mask.astype(bool)

def analyze_spectral_support(
    input_spectrum: np.ndarray,
    mask: np.ndarray,
    grid: Grid,
    wavelength: float,
    z: float,
) -> Dict[str, float]:
    """Analyze retained spectral components and energy under band-limiting."""
    total_elements = mask.size
    retained_elements = int(np.count_nonzero(mask))
    element_ratio = retained_elements / float(total_elements)

    spectral_power = np.abs(input_spectrum) ** 2
    total_energy = float(np.sum(spectral_power))
    if total_energy > 0.0:
        retained_energy = float(np.sum(spectral_power[mask]))
        energy_ratio = retained_energy / total_energy
    else:
        energy_ratio = 1.0

    fx_limit, fy_limit = compute_bandlimit_frequencies(wavelength, z, grid.Sx, grid.Sy)
    return {
        "fx_limit": float(fx_limit),
        "fy_limit": float(fy_limit),
        "retained_elements": retained_elements,
        "total_elements": total_elements,
        "element_ratio": float(element_ratio),
        "energy_ratio": float(energy_ratio),
    }
