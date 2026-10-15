from typing import Tuple, Optional, Dict
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from diffraction.grids import Grid, create_grid
from diffraction.bandlimit import make_bandlimit_mask

_TF_CACHE: Dict[Tuple, np.ndarray] = {}

def clear_transfer_function_cache():
    """Clear internal transfer function cache."""
    _TF_CACHE.clear()

def pad_field(field: np.ndarray, factor: int = 2) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    """Pad 2D field centrally by an integer factor."""
    if factor <= 1:
        ny, nx = field.shape
        return field.copy(), (slice(0, ny), slice(0, nx))

    ny, nx = field.shape
    new_ny = ny * factor
    new_nx = nx * factor
    padded = np.zeros((new_ny, new_nx), dtype=field.dtype)

    start_y = (new_ny - ny) // 2
    start_x = (new_nx - nx) // 2
    crop_slices = (slice(start_y, start_y + ny), slice(start_x, start_x + nx))
    padded[crop_slices] = field
    return padded, crop_slices

def unpad_field(field: np.ndarray, crop_slices: Tuple[slice, slice]) -> np.ndarray:
    """Crop padded field back to original dimensions."""
    return field[crop_slices]

def get_asm_transfer_function(
    grid: Grid,
    wavelength: float,
    z: float,
) -> np.ndarray:
    """Generate the exact Angular Spectrum transfer function H_AS."""
    if z == 0.0:
        return np.ones((grid.ny, grid.nx), dtype=np.complex128)

    k = 2.0 * np.pi / wavelength
    fsq = grid.FX**2 + grid.FY**2
    inv_lam_sq = (1.0 / wavelength) ** 2

    prop_mask = fsq <= inv_lam_sq
    evan_mask = ~prop_mask

    H = np.zeros((grid.ny, grid.nx), dtype=np.complex128)

    kz_prop = 2.0 * np.pi * np.sqrt(inv_lam_sq - fsq[prop_mask])
    H[prop_mask] = np.exp(1j * kz_prop * z)

    if np.any(evan_mask):
        alpha_evan = 2.0 * np.pi * np.sqrt(fsq[evan_mask] - inv_lam_sq)
        decay = alpha_evan * z
        valid_decay = decay < 300.0
        H_evan = np.zeros_like(decay)
        H_evan[valid_decay] = np.exp(-decay[valid_decay])
        H[evan_mask] = H_evan

    return H

def get_blas_transfer_function(
    grid: Grid,
    wavelength: float,
    z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the Band-Limited Angular Spectrum transfer function and its mask."""
    H_asm = get_asm_transfer_function(grid, wavelength, z)
    if z == 0.0:
        mask = np.ones((grid.ny, grid.nx), dtype=bool)
        return H_asm, mask

    mask = make_bandlimit_mask(grid, wavelength, z)
    H_blas = H_asm * mask
    return H_blas, mask

def get_fresnel_transfer_function(
    grid: Grid,
    wavelength: float,
    z: float,
) -> np.ndarray:
    """Generate the paraxial Fresnel transfer function H_F."""
    if z == 0.0:
        return np.ones((grid.ny, grid.nx), dtype=np.complex128)

    k = 2.0 * np.pi / wavelength
    fsq = grid.FX**2 + grid.FY**2
    phase = k * z - np.pi * wavelength * z * fsq
    return np.exp(1j * phase)

def propagate_field(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    method: str = "blas",
    padding: int = 2,
    return_transfer: bool = False,
) -> Tuple[np.ndarray, Grid, Optional[np.ndarray]]:
    """Propagate a complex optical field to distance z using FFT-based methods.
    
    Parameters
    ----------
    field : np.ndarray
        Complex field array of shape (Ny, Nx).
    wavelength : float
        Optical wavelength in meters.
    z : float
        Propagation distance in meters.
    dx, dy : float
        Spatial sampling intervals in meters.
    method : str
        One of 'asm', 'blas', 'fresnel'.
    padding : int
        Zero-padding factor (default 2).
    return_transfer : bool
        Whether to return the calculated transfer function.
    """
    if field.ndim != 2:
        raise ValueError(f"Field must be a 2D array, got shape {field.shape}")
    if z < 0.0:
        raise ValueError(f"Propagation distance must be non-negative, got z={z}")

    method_key = method.lower()
    if method_key not in ("asm", "blas", "fresnel"):
        raise ValueError(f"Unknown propagation method: {method}. Choose from 'asm', 'blas', 'fresnel'")

    ny, nx = field.shape
    padded_field, crop_slices = pad_field(field, factor=padding)
    ny_pad, nx_pad = padded_field.shape
    pad_grid = create_grid(nx=nx_pad, ny=ny_pad, dx=dx, dy=dy)

    cache_key = (method_key, nx_pad, ny_pad, dx, dy, wavelength, float(z))
    H = _TF_CACHE.get(cache_key)

    if H is None:
        if method_key == "asm":
            H = get_asm_transfer_function(pad_grid, wavelength, z)
        elif method_key == "blas":
            H, _ = get_blas_transfer_function(pad_grid, wavelength, z)
        elif method_key == "fresnel":
            H = get_fresnel_transfer_function(pad_grid, wavelength, z)
        _TF_CACHE[cache_key] = H

    U_in_spec = fftshift(fft2(ifftshift(padded_field)))
    U_out_spec = U_in_spec * H
    padded_out = fftshift(ifft2(ifftshift(U_out_spec)))

    out_field = unpad_field(padded_out, crop_slices)
    out_grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)

    transfer_return = H if return_transfer else None
    return out_field, out_grid, transfer_return

def propagate_asm(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    padding: int = 2,
    return_transfer: bool = False,
) -> Tuple[np.ndarray, Grid, Optional[np.ndarray]]:
    """Propagate field using conventional unband-limited Angular Spectrum."""
    return propagate_field(
        field=field,
        wavelength=wavelength,
        z=z,
        dx=dx,
        dy=dy,
        method="asm",
        padding=padding,
        return_transfer=return_transfer,
    )

def propagate_blas(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    padding: int = 2,
    return_transfer: bool = False,
) -> Tuple[np.ndarray, Grid, Optional[np.ndarray]]:
    """Propagate field using Band-Limited Angular Spectrum (Matsushima & Shimobaba 2009)."""
    return propagate_field(
        field=field,
        wavelength=wavelength,
        z=z,
        dx=dx,
        dy=dy,
        method="blas",
        padding=padding,
        return_transfer=return_transfer,
    )

def propagate_fresnel(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    padding: int = 2,
    return_transfer: bool = False,
) -> Tuple[np.ndarray, Grid, Optional[np.ndarray]]:
    """Propagate field using paraxial Fresnel transfer function."""
    return propagate_field(
        field=field,
        wavelength=wavelength,
        z=z,
        dx=dx,
        dy=dy,
        method="fresnel",
        padding=padding,
        return_transfer=return_transfer,
    )

def propagate_direct_integration(
    field: np.ndarray,
    wavelength: float,
    z: float,
    dx: float,
    dy: float,
    x_out: np.ndarray,
    y_out: np.ndarray,
) -> np.ndarray:
    """Direct numerical integration of Rayleigh-Sommerfeld integral.
    
    Used as an independent non-FFT ground truth reference on small grids.
    """
    if z <= 0.0:
        raise ValueError(f"Direct integration requires z > 0, got z={z}")

    ny_in, nx_in = field.shape
    in_grid = create_grid(nx=nx_in, ny=ny_in, dx=dx, dy=dy)
    k = 2.0 * np.pi / wavelength

    active_indices = np.where(np.abs(field) > 1e-12)
    active_x = in_grid.X[active_indices]
    active_y = in_grid.Y[active_indices]
    active_val = field[active_indices]

    dA = dx * dy
    nx_out = len(x_out)
    ny_out = len(y_out)
    out_field = np.zeros((ny_out, nx_out), dtype=np.complex128)

    for iy, yo in enumerate(y_out):
        for ix, xo in enumerate(x_out):
            rx = xo - active_x
            ry = yo - active_y
            r = np.sqrt(rx**2 + ry**2 + z**2)
            G = (z / (2.0 * np.pi * r**2)) * np.exp(1j * k * r) * (1.0 / r - 1j * k)
            out_field[iy, ix] = np.sum(active_val * G) * dA

    return out_field
