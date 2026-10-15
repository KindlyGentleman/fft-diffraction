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
    propagate_sas,
    propagate_sft_fresnel,
    propagate_rs_direct,
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    relative_l2_error,
    total_power,
    radial_profile,
    find_first_radial_minimum,
    compute_bandlimit_frequencies,
    double_slit_aperture,
    annular_aperture,
    gaussian_field,
    binary_grating,
    triangular_aperture,
    shifted_circular_aperture,
    compute_shared_region_l2_error,
    sas_output_scaling,
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


def test_sas_zero_distance_identity():
    """Verify that SAS propagation at z=0 reproduces input field to machine precision."""
    grid = create_grid(nx=128, ny=128, dx=5e-6, dy=5e-6)
    u_in = circular_aperture(grid, radius=1.5e-4)
    wavelength = 532e-9

    u_sas, grid_out = propagate_sas(u_in, wavelength, z=0.0, dx=grid.dx, dy=grid.dy, padding=2)

    assert relative_l2_error(u_sas, u_in) < 1e-14
    assert np.isclose(grid_out.dx, grid.dx)
    assert np.isclose(grid_out.dy, grid.dy)


def test_sas_circular_fraunhofer_airy():
    """Verify Scalable Angular Spectrum against analytical Airy diffraction on scaled grid."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4
    z = 0.50

    grid_in = create_grid(nx, ny, dx, dy)
    u_in = circular_aperture(grid_in, radius=radius)
    u_sas, grid_out = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    i_sas = np.abs(u_sas)**2
    i_ana = airy_pattern(grid_out, wavelength, z=z, radius=radius)

    # Check output grid pitch formula: dq = lambda * z / (2 * N * dx)
    expected_dq = wavelength * z / (2 * nx * dx)
    assert np.isclose(grid_out.dx, expected_dq)

    # Relative L2 intensity error compared with analytical Airy pattern
    p_sas_n = i_sas / np.max(i_sas)
    p_ana_n = i_ana / np.max(i_ana)
    rel_err = np.linalg.norm(p_sas_n - p_ana_n) / np.linalg.norm(p_ana_n)
    assert rel_err < 0.015  # Measured ~0.67%

    # Verify first dark ring position on scaled radial profile
    r_num, prof_num = radial_profile(i_sas, grid_out.x, grid_out.y, num_bins=120)
    r1_theory = airy_first_zero(wavelength, z, radius)
    r1_meas = find_first_radial_minimum(r_num, prof_num, search_range=(0.7 * r1_theory, 1.4 * r1_theory))
    assert abs(r1_meas - r1_theory) / r1_theory < 0.035  # Measured ~2.6% at N=256


def test_sas_rectangular_fraunhofer_sinc():
    """Verify Scalable Angular Spectrum rectangular diffraction against analytical sinc pattern."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    width, height = 3.0e-4, 4.5e-4
    z = 0.50

    grid_in = create_grid(nx, ny, dx, dy)
    u_in = rectangular_aperture(grid_in, width=width, height=height)
    u_sas, grid_out = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    i_sas = np.abs(u_sas)**2
    i_ana = sinc_pattern(grid_out, wavelength, z=z, width=width, height=height)

    cy, cx = ny // 2, nx // 2
    x_num = i_sas[cy, :] / np.max(i_sas[cy, :])
    x_ana = i_ana[cy, :] / np.max(i_ana[cy, :])
    y_num = i_sas[:, cx] / np.max(i_sas[:, cx])
    y_ana = i_ana[:, cx] / np.max(i_ana[:, cx])

    err_x = np.linalg.norm(x_num - x_ana) / np.linalg.norm(x_ana)
    err_y = np.linalg.norm(y_num - y_ana) / np.linalg.norm(y_ana)

    assert err_x < 0.03  # Measured ~1.4%
    assert err_y < 0.05  # Measured ~2.75%


def test_sas_vs_single_step_fresnel_near_field():
    """Verify that SAS differs from single-step Fresnel when non-paraxial correction is active."""
    nx, ny = 128, 128
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.0e-4
    z = 0.05

    grid_in = create_grid(nx, ny, dx, dy)
    u_in = circular_aperture(grid_in, radius=radius)

    u_sas, _ = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
    u_fr, _ = propagate_sft_fresnel(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    diff = np.linalg.norm(u_sas - u_fr) / np.linalg.norm(u_sas)
    # The non-paraxial precompensation introduces a measurable phase/amplitude correction
    assert diff > 1e-4


def test_aperture_gallery_masks():
    """Verify that all eight aperture generation functions produce valid, non-zero fields."""
    grid = create_grid(nx=256, ny=256, dx=5e-6, dy=5e-6)

    # 1. Circular
    u_circ = circular_aperture(grid, radius=1.5e-4)
    assert np.all((u_circ == 0.0) | (u_circ == 1.0))
    assert np.sum(np.abs(u_circ)) > 0

    # 2. Rectangular
    u_rect = rectangular_aperture(grid, width=3.0e-4, height=4.5e-4)
    assert np.all((u_rect == 0.0) | (u_rect == 1.0))
    assert np.sum(np.abs(u_rect)) > 0

    # 3. Double slit
    u_slit = double_slit_aperture(grid, slit_width=5e-5, slit_height=4e-4, separation=2e-4)
    assert np.all((u_slit == 0.0) | (u_slit == 1.0))
    # Check left-right coordinate symmetry
    assert np.isclose(np.sum(u_slit[grid.X < 0]), np.sum(u_slit[grid.X > 0]))

    # 4. Annular
    u_ann = annular_aperture(grid, r_inner=8e-5, r_outer=1.8e-4)
    assert np.all((u_ann == 0.0) | (u_ann == 1.0))
    # Origin should be dark (inner obscuration)
    assert u_ann[grid.ny // 2, grid.nx // 2] == 0.0

    # 5. Gaussian field
    u_gauss = gaussian_field(grid, waist=1.5e-4)
    assert np.isclose(np.max(np.abs(u_gauss)), 1.0)
    assert np.all(np.abs(u_gauss) >= 0.0)

    # 6. Binary grating
    u_grat = binary_grating(grid, period=6e-5, duty_cycle=0.5, width=5e-4, height=4e-4)
    assert np.all((u_grat == 0.0) | (u_grat == 1.0))
    assert np.sum(np.abs(u_grat)) > 0

    # 7. Triangular
    u_tri = triangular_aperture(grid, side_length=3.5e-4)
    assert np.all((u_tri == 0.0) | (u_tri == 1.0))
    assert np.sum(np.abs(u_tri)) > 0
    # Left-right coordinate mirror symmetry
    assert np.isclose(np.sum(u_tri[grid.X < 0]), np.sum(u_tri[grid.X > 0]))

    # 8. Shifted circular
    u_shift = shifted_circular_aperture(grid, radius=1.2e-4, x_offset=2.5e-4, y_offset=2.5e-4)
    assert np.all((u_shift == 0.0) | (u_shift == 1.0))
    # Center origin should be dark because circle is shifted away
    assert u_shift[grid.ny // 2, grid.nx // 2] == 0.0


def test_shared_region_l2_error():
    """Verify that compute_shared_region_l2_error correctly restricts evaluation."""
    grid = create_grid(nx=128, ny=128, dx=10e-6, dy=10e-6)
    arr1 = np.ones((128, 128))
    arr2 = np.ones((128, 128))

    err_zero = compute_shared_region_l2_error(arr1, grid, arr2, x_max=2e-4, y_max=2e-4)
    assert np.isclose(err_zero, 0.0)

    # Difference placed only outside the shared region
    arr2_out = arr2.copy()
    arr2_out[0, 0] = 99.0
    err_outside = compute_shared_region_l2_error(arr1, grid, arr2_out, x_max=2e-4, y_max=2e-4)
    assert np.isclose(err_outside, 0.0)

    # Difference inside the shared region
    arr2_in = arr2.copy()
    arr2_in[64, 64] = 2.0
    err_inside = compute_shared_region_l2_error(arr1, grid, arr2_in, x_max=2e-4, y_max=2e-4)
    assert err_inside > 0.0


def test_sas_output_scaling():
    """Verify analytical formula for SAS destination grid scaling."""
    wavelength = 532e-9
    z = 0.50
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    padding = 2

    dqx, dqy, Sx_out, Sy_out = sas_output_scaling(wavelength, z, nx, ny, dx, dy, padding=padding)
    expected_dq = wavelength * z / (nx * padding * dx)
    assert np.isclose(dqx, expected_dq)
    assert np.isclose(dqy, expected_dq)
    assert np.isclose(Sx_out, nx * expected_dq)
    assert np.isclose(Sy_out, ny * expected_dq)


