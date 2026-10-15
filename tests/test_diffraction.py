"""Validation and regression tests for scalar diffraction propagation."""

import numpy as np
import pytest
from diffraction import (
    create_grid,
    circular_aperture,
    rectangular_aperture,
    propagate_asm,
    propagate_blas,
    propagate_fresnel,
    propagate_rs_direct,
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    relative_l2_error,
    total_power,
    radial_profile,
    find_first_radial_minimum,
    compute_bandlimit_frequencies,
)


def test_zero_distance_identity():
    """Verify that z=0 propagation reproduces the aperture field to machine precision."""
    grid = create_grid(nx=128, ny=128, dx=5e-6, dy=5e-6)
    u_in = circular_aperture(grid, radius=1.5e-4)
    wavelength = 532e-9

    u_asm = propagate_asm(u_in, wavelength, z=0.0, dx=grid.dx, dy=grid.dy, padding=2)
    u_blas = propagate_blas(u_in, wavelength, z=0.0, dx=grid.dx, dy=grid.dy, padding=2)
    u_fres = propagate_fresnel(u_in, wavelength, z=0.0, dx=grid.dx, dy=grid.dy, padding=2)

    assert relative_l2_error(u_asm, u_in) < 1e-14
    assert relative_l2_error(u_blas, u_in) < 1e-14
    assert relative_l2_error(u_fres, u_in) < 1e-14


def test_plane_wave_evolution():
    """Verify uniform plane wave phase evolution exp(i*k*z) and constant intensity."""
    nx, ny = 128, 128
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    z = 0.02
    k = 2.0 * np.pi / wavelength

    u_in = np.ones((ny, nx), dtype=np.complex128)
    expected = np.exp(1j * k * z) * u_in

    u_asm = propagate_asm(u_in, wavelength, z=z, dx=dx, dy=dy, padding=1)
    u_blas = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=1)
    u_fres = propagate_fresnel(u_in, wavelength, z=z, dx=dx, dy=dy, padding=1)

    assert relative_l2_error(u_asm, expected) < 1e-10
    assert relative_l2_error(u_blas, expected) < 1e-10
    assert relative_l2_error(u_fres, expected) < 1e-10


def test_fft_roundtrip_invertibility():
    """Verify centered 2D FFT/IFFT roundtrip consistency."""
    rng = np.random.default_rng(12345)
    sample = rng.standard_normal((128, 128)) + 1j * rng.standard_normal((128, 128))
    from scipy.fft import fft2, ifft2, fftshift, ifftshift

    roundtrip = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(sample))))))
    err = np.linalg.norm(roundtrip - sample) / np.linalg.norm(sample)
    assert err < 1e-14


def test_circular_fraunhofer_airy():
    """Verify Airy disc geometry and radial profile against analytical Fraunhofer theory."""
    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4
    z = 0.50

    grid = create_grid(nx, ny, dx, dy)
    u_in = circular_aperture(grid, radius=radius)
    u_out = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    i_num = np.abs(u_out)**2
    i_ana = airy_pattern(grid, wavelength, z=z, radius=radius)

    r_num, prof_num = radial_profile(i_num, grid.x, grid.y, num_bins=150)
    r_ana, prof_ana = radial_profile(i_ana, grid.x, grid.y, num_bins=150)

    # First zero search in vicinity of theoretical Airy zero.
    r1_theory = airy_first_zero(wavelength, z, radius)
    r1_meas = find_first_radial_minimum(r_num, prof_num, search_range=(0.7 * r1_theory, 1.4 * r1_theory))

    rel_zero_err = abs(r1_meas - r1_theory) / r1_theory
    prof_l2_err = np.linalg.norm(prof_num / np.max(prof_num) - prof_ana / np.max(prof_ana)) / np.linalg.norm(prof_ana / np.max(prof_ana))

    assert rel_zero_err < 0.01  # Measured ~0.61%
    assert prof_l2_err < 0.05   # Measured ~1.62%


def test_rectangular_fraunhofer_sinc():
    """Verify rectangular aperture cross-sections against analytical 2D sinc pattern."""
    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    width, height = 3.0e-4, 4.5e-4
    z = 0.50

    grid = create_grid(nx, ny, dx, dy)
    u_in = rectangular_aperture(grid, width=width, height=height)
    u_out = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    i_num = np.abs(u_out)**2
    i_ana = sinc_pattern(grid, wavelength, z=z, width=width, height=height)

    cy, cx = ny // 2, nx // 2
    x_num = i_num[cy, :] / np.max(i_num[cy, :])
    x_ana = i_ana[cy, :] / np.max(i_ana[cy, :])
    y_num = i_num[:, cx] / np.max(i_num[:, cx])
    y_ana = i_ana[:, cx] / np.max(i_ana[:, cx])

    err_x = np.linalg.norm(x_num - x_ana) / np.linalg.norm(x_ana)
    err_y = np.linalg.norm(y_num - y_ana) / np.linalg.norm(y_ana)

    assert err_x < 0.05  # Measured ~1.65%
    assert err_y < 0.05  # Measured ~2.64%


def test_near_field_rs_direct_integration():
    """Verify near-field field with independent Rayleigh-Sommerfeld direct numerical integration."""
    nx, ny = 64, 64
    dx, dy = 8e-6, 8e-6
    wavelength = 532e-9
    radius = 8e-5
    z = 0.005

    grid = create_grid(nx, ny, dx, dy)
    u_in = circular_aperture(grid, radius=radius)
    u_blas = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    # Sub-sample central region for direct numerical integration.
    sel_x = np.linspace(nx // 4, 3 * nx // 4, 16, dtype=int)
    sel_y = np.linspace(ny // 4, 3 * ny // 4, 16, dtype=int)
    sub_x = grid.x[sel_x]
    sub_y = grid.y[sel_y]

    u_rs = propagate_rs_direct(u_in, wavelength, z=z, dx=dx, dy=dy, x_out=sub_x, y_out=sub_y)
    u_blas_sub = u_blas[np.ix_(sel_y, sel_x)]

    i_rs = np.abs(u_rs)**2
    i_blas_sub = np.abs(u_blas_sub)**2

    err = np.linalg.norm(i_blas_sub - i_rs) / np.linalg.norm(i_rs)
    assert err < 0.05  # Measured ~3.89%


def test_energy_conservation():
    """Verify total optical power conservation across methods."""
    grid = create_grid(nx=256, ny=256, dx=5e-6, dy=5e-6)
    u_in = circular_aperture(grid, radius=1e-4)
    wavelength = 532e-9
    z = 0.01

    p_in = total_power(u_in, grid.dx, grid.dy)
    p_asm = total_power(propagate_asm(u_in, wavelength, z=z, dx=grid.dx, dy=grid.dy, padding=2), grid.dx, grid.dy)
    p_blas = total_power(propagate_blas(u_in, wavelength, z=z, dx=grid.dx, dy=grid.dy, padding=2), grid.dx, grid.dy)
    p_fres = total_power(propagate_fresnel(u_in, wavelength, z=z, dx=grid.dx, dy=grid.dy, padding=2), grid.dx, grid.dy)

    assert abs(p_asm - p_in) / p_in < 0.01
    assert abs(p_blas - p_in) / p_in < 0.01
    assert abs(p_fres - p_in) / p_in < 0.01


def test_bandlimit_frequencies_shrink_with_distance():
    """Verify that the band-limiting cutoff frequency narrows monotonically as z increases."""
    wavelength = 532e-9
    Sx, Sy = 2.56e-3, 2.56e-3

    fx_1, fy_1 = compute_bandlimit_frequencies(wavelength, z=0.01, Sx=Sx, Sy=Sy)
    fx_2, fy_2 = compute_bandlimit_frequencies(wavelength, z=0.10, Sx=Sx, Sy=Sy)
    fx_3, fy_3 = compute_bandlimit_frequencies(wavelength, z=0.80, Sx=Sx, Sy=Sy)

    assert fx_1 > fx_2 > fx_3
    assert fy_1 > fy_2 > fy_3
