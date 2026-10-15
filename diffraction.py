"""Scalar optical diffraction propagation methods and validation utilities.

Implements Fresnel, conventional Angular Spectrum (ASM), and Band-Limited
Angular Spectrum (BL-ASM) based on Matsushima and Shimobaba (2009).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1


@dataclass(frozen=True)
class Grid:
    """Two-dimensional spatial and spatial-frequency grid representation."""
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    FX: np.ndarray
    FY: np.ndarray
    dx: float
    dy: float
    nx: int
    ny: int
    Sx: float
    Sy: float


def create_grid(nx: int, ny: int, dx: float, dy: float) -> Grid:
    """Generate centered spatial and spatial-frequency grids."""
    x = (np.arange(nx) - nx / 2.0) * dx
    y = (np.arange(ny) - ny / 2.0) * dy
    X, Y = np.meshgrid(x, y)

    fx = (np.arange(nx) - nx / 2.0) / (nx * dx)
    fy = (np.arange(ny) - ny / 2.0) / (ny * dy)
    FX, FY = np.meshgrid(fx, fy)

    return Grid(
        x=x, y=y, X=X, Y=Y,
        fx=fx, fy=fy, FX=FX, FY=FY,
        dx=dx, dy=dy, nx=nx, ny=ny,
        Sx=nx * dx, Sy=ny * dy
    )


def circular_aperture(grid: Grid, radius: float) -> np.ndarray:
    """Generate a binary circular aperture mask."""
    r = np.sqrt(grid.X**2 + grid.Y**2)
    return (r <= radius).astype(np.complex128)


def rectangular_aperture(grid: Grid, width: float, height: float) -> np.ndarray:
    """Generate a binary rectangular aperture mask."""
    mask = (np.abs(grid.X) <= width / 2.0) & (np.abs(grid.Y) <= height / 2.0)
    return mask.astype(np.complex128)


def pad_field(field: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    """Centrally zero-pad a 2D field by an integer factor."""
    if factor <= 1:
        ny, nx = field.shape
        return field.copy(), (slice(0, ny), slice(0, nx))

    ny, nx = field.shape
    new_ny, new_nx = ny * factor, nx * factor
    padded = np.zeros((new_ny, new_nx), dtype=field.dtype)

    sy = (new_ny - ny) // 2
    sx = (new_nx - nx) // 2
    slices = (slice(sy, sy + ny), slice(sx, sx + nx))
    padded[slices] = field
    return padded, slices


def compute_bandlimit_frequencies(wavelength: float, z: float, Sx: float, Sy: float) -> Tuple[float, float]:
    """Calculate Matsushima-Shimobaba critical cutoff frequencies.

    Reference: Matsushima & Shimobaba, Opt. Express 17(22), 19662 (2009).
    """
    if z == 0.0:
        return 1.0 / wavelength, 1.0 / wavelength

    fx_lim = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * z / Sx) ** 2))
    fy_lim = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * z / Sy) ** 2))
    return fx_lim, fy_lim


def asm_transfer_function(grid: Grid, wavelength: float, z: float) -> np.ndarray:
    """Calculate the exact free-space Angular Spectrum transfer function."""
    if z == 0.0:
        return np.ones((grid.ny, grid.nx), dtype=np.complex128)

    fsq = grid.FX**2 + grid.FY**2
    inv_lam_sq = (1.0 / wavelength) ** 2
    prop_mask = fsq <= inv_lam_sq

    H = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    kz_prop = 2.0 * np.pi * np.sqrt(inv_lam_sq - fsq[prop_mask])
    H[prop_mask] = np.exp(1j * kz_prop * z)

    # Evanescent components decay exponentially for z > 0.
    evan_mask = ~prop_mask
    if np.any(evan_mask):
        alpha = 2.0 * np.pi * np.sqrt(fsq[evan_mask] - inv_lam_sq)
        decay = alpha * z
        valid = decay < 300.0
        H_evan = np.zeros_like(decay)
        H_evan[valid] = np.exp(-decay[valid])
        H[evan_mask] = H_evan

    return H


def blas_transfer_function(grid: Grid, wavelength: float, z: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the band-limited Angular Spectrum transfer function and its mask."""
    H = asm_transfer_function(grid, wavelength, z)
    if z == 0.0:
        mask = np.ones((grid.ny, grid.nx), dtype=bool)
        return H, mask

    fx_lim, fy_lim = compute_bandlimit_frequencies(wavelength, z, grid.Sx, grid.Sy)
    mask = (np.abs(grid.FX) <= fx_lim) & (np.abs(grid.FY) <= fy_lim)
    return H * mask, mask


def fresnel_transfer_function(grid: Grid, wavelength: float, z: float) -> np.ndarray:
    """Calculate the paraxial Fresnel transfer function."""
    if z == 0.0:
        return np.ones((grid.ny, grid.nx), dtype=np.complex128)

    k = 2.0 * np.pi / wavelength
    fsq = grid.FX**2 + grid.FY**2
    return np.exp(1j * (k * z - np.pi * wavelength * z * fsq))


def propagate(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    method: str = "blas",
    padding: int = 2,
) -> np.ndarray:
    """Propagate a 2D optical field using an FFT-based transfer function.

    Parameters:
        field: Complex input optical field.
        wavelength: Light wavelength in meters.
        z: Propagation distance in meters.
        dx, dy: Pixel sampling pitches in meters.
        method: Propagation method ('asm', 'blas', 'fresnel').
        padding: Zero-padding factor for linear convolution.
    """
    if field.ndim != 2:
        raise ValueError("Field array must be 2D.")
    if z < 0.0:
        raise ValueError("Propagation distance must be non-negative.")

    method_key = method.lower()
    if method_key not in ("asm", "blas", "fresnel"):
        raise ValueError(f"Unknown propagation method: {method}")

    padded, crop_slices = pad_field(field, factor=padding)
    ny_p, nx_p = padded.shape
    pgrid = create_grid(nx=nx_p, ny=ny_p, dx=dx, dy=dy)

    if method_key == "asm":
        H = asm_transfer_function(pgrid, wavelength, z)
    elif method_key == "blas":
        H, _ = blas_transfer_function(pgrid, wavelength, z)
    else:
        H = fresnel_transfer_function(pgrid, wavelength, z)

    spec_in = fftshift(fft2(ifftshift(padded)))
    spec_out = spec_in * H
    padded_out = fftshift(ifft2(ifftshift(spec_out)))

    return padded_out[crop_slices]


def propagate_asm(field: np.ndarray, wavelength: float, z: float, dx: float, dy: float, padding: int = 2) -> np.ndarray:
    """Propagate field with standard Angular Spectrum."""
    return propagate(field, wavelength, z, dx, dy, method="asm", padding=padding)


def propagate_blas(field: np.ndarray, wavelength: float, z: float, dx: float, dy: float, padding: int = 2) -> np.ndarray:
    """Propagate field with Band-Limited Angular Spectrum."""
    return propagate(field, wavelength, z, dx, dy, method="blas", padding=padding)


def propagate_fresnel(field: np.ndarray, wavelength: float, z: float, dx: float, dy: float, padding: int = 2) -> np.ndarray:
    """Propagate field with Fresnel transfer function."""
    return propagate(field, wavelength, z, dx, dy, method="fresnel", padding=padding)


def propagate_rs_direct(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    x_out: np.ndarray,
    y_out: np.ndarray,
) -> np.ndarray:
    """Direct numerical integration of the first Rayleigh-Sommerfeld diffraction integral."""
    if z <= 0.0:
        raise ValueError("Direct integration requires z > 0.")

    ny_in, nx_in = field.shape
    in_grid = create_grid(nx=nx_in, ny=ny_in, dx=dx, dy=dy)
    k = 2.0 * np.pi / wavelength

    active = np.abs(field) > 1e-12
    act_x = in_grid.X[active]
    act_y = in_grid.Y[active]
    act_val = field[active]

    dA = dx * dy
    out_field = np.zeros((len(y_out), len(x_out)), dtype=np.complex128)

    for iy, yo in enumerate(y_out):
        for ix, xo in enumerate(x_out):
            r = np.sqrt((xo - act_x)**2 + (yo - act_y)**2 + z**2)
            # Sommerfeld radiation kernel with obliquity factor cos(theta) = z / r.
            kernel = (z / (2.0 * np.pi * r**2)) * np.exp(1j * k * r) * (1.0 / r - 1j * k)
            out_field[iy, ix] = np.sum(act_val * kernel) * dA

    return out_field


def jinc(q: np.ndarray) -> np.ndarray:
    """Evaluate 2 * J1(q) / q, handling q=0 continuously."""
    out = np.ones_like(q, dtype=np.float64)
    nz = np.abs(q) > 1e-12
    out[nz] = 2.0 * j1(q[nz]) / q[nz]
    return out


def airy_first_zero(wavelength: float, z: float, radius: float) -> float:
    """Theoretical radius of the first dark Airy ring in meters."""
    k = 2.0 * np.pi / wavelength
    return float(3.8317059702 * z / (k * radius))


def airy_pattern(grid: Grid, wavelength: float, z: float, radius: float) -> np.ndarray:
    """Calculate analytical Fraunhofer intensity pattern for a circular aperture."""
    k = 2.0 * np.pi / wavelength
    r = np.sqrt(grid.X**2 + grid.Y**2)
    q = k * radius * r / z
    aperture_area = np.pi * (radius**2)
    amplitude = aperture_area / (wavelength * z) * jinc(q)
    return amplitude**2


def sinc_pattern(grid: Grid, wavelength: float, z: float, width: float, height: float) -> np.ndarray:
    """Calculate analytical Fraunhofer intensity pattern for a rectangular aperture."""
    sinc_x = np.sinc(width * grid.X / (wavelength * z))
    sinc_y = np.sinc(height * grid.Y / (wavelength * z))
    amplitude = (width * height) / (wavelength * z) * sinc_x * sinc_y
    return amplitude**2


def relative_l2_error(u_test: np.ndarray, u_ref: np.ndarray) -> float:
    """Calculate relative L2 error norm between two field or intensity arrays."""
    norm_ref = np.linalg.norm(u_ref)
    if norm_ref == 0.0:
        return float(np.linalg.norm(u_test))
    return float(np.linalg.norm(u_test - u_ref) / norm_ref)


def total_power(field: np.ndarray, dx: float, dy: float) -> float:
    """Integrated squared magnitude in arbitrary optical power units (a.u.)."""
    return float(np.sum(np.abs(field)**2) * dx * dy)


def radial_profile(intensity: np.ndarray, x: np.ndarray, y: np.ndarray, num_bins: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally averaged radial profile from the origin."""
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    r_max = min(np.max(np.abs(x)), np.max(np.abs(y)))

    bin_edges = np.linspace(0, r_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.digitize(R.ravel(), bin_edges) - 1

    profile = np.zeros(num_bins, dtype=np.float64)
    flat_int = intensity.ravel()
    for b in range(num_bins):
        mask = indices == b
        if np.any(mask):
            profile[b] = np.mean(flat_int[mask])

    return bin_centers, profile


def find_first_radial_minimum(
    r_coords: np.ndarray,
    radial_prof: np.ndarray,
    search_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Find the radial position of the first intensity minimum (dark ring)."""
    if search_range is not None:
        r_min, r_max = search_range
        valid = (r_coords >= r_min) & (r_coords <= r_max)
        r_sub = r_coords[valid]
        p_sub = radial_prof[valid]
    else:
        r_sub = r_coords
        p_sub = radial_prof

    if len(p_sub) < 3:
        return float(r_sub[np.argmin(p_sub)])

    diffs = np.diff(p_sub)
    sign_changes = np.where((diffs[:-1] <= 0) & (diffs[1:] > 0))[0]
    if len(sign_changes) > 0:
        return float(r_sub[sign_changes[0] + 1])
    return float(r_sub[np.argmin(p_sub)])

