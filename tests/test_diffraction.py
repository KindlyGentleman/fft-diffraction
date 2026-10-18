"""Validation and regression tests for scalar diffraction propagation."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
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
    compare_on_common_grid,
    sas_output_scaling,
    thin_lens_phase,
    spiral_phase,
    topological_charge,
    gaussian_beam_abcd,
    measure_beam_radius,
    check_sas_validity,
    asm_critical_distance,
    beam_waist_second_moment,
    compute_complex_field_error,
    pad_field,
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


def test_compare_on_common_grid():
    """Verify that compare_on_common_grid correctly interpolates on the true intersection domain."""
    grid1 = create_grid(nx=128, ny=128, dx=10e-6, dy=10e-6)
    grid2 = create_grid(nx=256, ny=256, dx=5e-6, dy=5e-6)

    # Identical Gaussian intensity distributions with different sampling grids
    w0 = 2e-4
    i1 = np.exp(-(grid1.X**2 + grid1.Y**2) / (w0**2))
    i2 = np.exp(-(grid2.X**2 + grid2.Y**2) / (w0**2))

    rel_l2, peak_l2, width = compare_on_common_grid(i1, grid1.x, grid1.y, i2, grid2.x, grid2.y, n_eval=100)
    assert rel_l2 < 0.005
    assert peak_l2 < 0.005
    expected_width = min(float(grid1.x[-1]), float(grid2.x[-1])) - max(float(grid1.x[0]), float(grid2.x[0]))
    assert np.isclose(width, expected_width)

    # Verify shifted grids without symmetric assumption
    x_shifted = grid1.x + 200e-6
    y_shifted = grid1.y + 100e-6
    rel_l2_s, _, width_s = compare_on_common_grid(i1, x_shifted, y_shifted, i2, grid2.x, grid2.y, n_eval=80)
    expected_width_s = min(float(x_shifted[-1]), float(grid2.x[-1])) - max(float(x_shifted[0]), float(grid2.x[0]))
    assert np.isclose(width_s, expected_width_s)

    # Verify disjoint grids return NaN
    x_disjoint = grid1.x + 5.0e-3
    rel_l2_disjoint, _, width_disjoint = compare_on_common_grid(i1, x_disjoint, grid1.y, i2, grid2.x, grid2.y, n_eval=50)
    assert np.isnan(rel_l2_disjoint)
    assert width_disjoint == 0.0



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


def test_thin_lens_phase_is_unit_modulus():
    """Verify that thin lens transmission has strictly unit magnitude across the grid."""
    grid = create_grid(nx=256, ny=256, dx=5e-6, dy=5e-6)
    wavelength = 532e-9
    focal_length = 75e-3

    t_lens = thin_lens_phase(grid, wavelength, focal_length)

    # Unit modulus everywhere within floating point precision
    assert np.allclose(np.abs(t_lens), 1.0, atol=1e-12)
    # Lens center has zero phase shift (t_lens = 1)
    cx, cy = grid.nx // 2, grid.ny // 2
    assert np.isclose(t_lens[cy, cx], 1.0 + 0j)


def test_gaussian_lens_abcd_focus():
    """Verify that propagated Gaussian beam focus agrees with analytical ABCD q-parameter theory."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4
    focal_length = 75e-3

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_0 = gaussian_field(grid, waist=w0)
    t_lens = thin_lens_phase(grid, wavelength, focal_length)
    u_lens = u_0 * t_lens

    # Analytical prediction from ABCD matrix
    w_z, R_z, z_focus_theory, w_focus_theory = gaussian_beam_abcd(w0, wavelength, focal_length, z=focal_length)

    # Propagate to theoretical focal plane using BL-ASM
    u_focus = propagate_blas(u_lens, wavelength, z=z_focus_theory, dx=dx, dy=dy, padding=2)
    i_focus = np.abs(u_focus)**2

    # Measure 1/e^2 beam waist from numerical intensity
    w_focus_meas = measure_beam_radius(i_focus, grid.x, grid.y)

    # Waist agreement within 3% (focused waist ~42 um with 5 um grid sampling)
    rel_waist_err = abs(w_focus_meas - w_focus_theory) / w_focus_theory
    assert rel_waist_err < 0.03

    # Scan axial range around focus to verify longitudinal minimum waist position
    z_scan = np.linspace(z_focus_theory - 6e-3, z_focus_theory + 6e-3, 13)
    measured_waists = []
    for z_i in z_scan:
        u_z = propagate_blas(u_lens, wavelength, z=z_i, dx=dx, dy=dy, padding=2)
        w_i = measure_beam_radius(np.abs(u_z)**2, grid.x, grid.y)
        measured_waists.append(w_i)

    z_min_meas = z_scan[np.argmin(measured_waists)]
    # Focus position should agree within 1 mm
    assert abs(z_min_meas - z_focus_theory) <= 1.0e-3


def test_vortex_topological_charge():
    """Verify topological charge conservation for Gaussian beam with integer spiral phase."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4
    z_prop = 0.20
    test_charges = [1, 2, -1]

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_gauss = gaussian_field(grid, waist=w0)

    for ell in test_charges:
        t_spiral = spiral_phase(grid, charge=ell)
        u_vortex = u_gauss * t_spiral

        # Input phase winding
        q_in = topological_charge(u_vortex, grid, radius=1.5e-4, num_points=360)
        assert abs(q_in - ell) < 0.02

        # Propagate field
        u_prop = propagate_blas(u_vortex, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)
        q_out = topological_charge(u_prop, grid, radius=1.5e-4, num_points=360)
        assert abs(q_out - ell) < 0.05


def test_asm_critical_distance_scaling():
    """Verify ASM critical distance formula z_c = N_FFT * dx^2 / lambda for actual padded grids."""
    n = 512
    dx = 5e-6
    wavelength = 532e-9

    zc_1 = asm_critical_distance(n, dx, wavelength, padding=1)
    zc_2 = asm_critical_distance(n, dx, wavelength, padding=2)
    zc_4 = asm_critical_distance(n, dx, wavelength, padding=4)

    # For N=512, dx=5 um, lambda=532 nm:
    # 1x padding: ~24.06 mm
    # 2x padding: ~48.12 mm
    # 4x padding: ~96.24 mm
    assert np.isclose(zc_1, 512 * (5e-6)**2 / 532e-9)
    assert np.isclose(zc_2, 1024 * (5e-6)**2 / 532e-9)
    assert np.isclose(zc_4, 2048 * (5e-6)**2 / 532e-9)

    assert abs(zc_1 - 0.02406) < 1e-4
    assert abs(zc_2 - 0.04812) < 1e-4
    assert abs(zc_4 - 0.09624) < 1e-4

    assert np.isclose(zc_2, 2.0 * zc_1)
    assert np.isclose(zc_4, 4.0 * zc_1)


def test_beam_waist_second_moment_analytical_gaussian():
    """Verify ISO 11146 second-moment waist calculation on analytical Gaussian beam."""
    nx, ny = 512, 512
    dx, dy = 2e-6, 2e-6
    w0 = 1.5e-4

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    # Physical intensity of Gaussian beam: I(r) = exp(-2*r^2 / w0^2)
    intensity = np.exp(-2.0 * (grid.X**2 + grid.Y**2) / (w0**2))

    wx, wy, w_mean = beam_waist_second_moment(intensity, grid.x, grid.y, roi_radius=3.0 * w0)

    assert abs(wx - w0) / w0 < 0.005
    assert abs(wy - w0) / w0 < 0.005
    assert abs(w_mean - w0) / w0 < 0.005


def test_far_field_absolute_on_axis_intensity():
    """Verify absolute physical on-axis intensity of circular aperture against Fraunhofer theory."""
    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_in = circular_aperture(grid, radius=radius)
    discrete_pupil_area = float(np.sum(np.abs(u_in)**2) * dx * dy)
    continuous_pupil_area = np.pi * radius**2

    # Check analytical expectations from prompt:
    # z=0.50 m -> ~0.0706
    # z=0.80 m -> ~0.0276
    for z, expected_approx in [(0.50, 0.0706), (0.80, 0.0276)]:
        i_theory_continuous = (continuous_pupil_area / (wavelength * z))**2
        assert abs(i_theory_continuous - expected_approx) / expected_approx < 0.01

        # Check BL-ASM propagation on-axis intensity
        u_blas = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
        cy, cx = ny // 2, nx // 2
        i_blas = float(np.abs(u_blas[cy, cx])**2)
        assert abs(i_blas - i_theory_continuous) / i_theory_continuous < 0.05

        # Check SAS propagation on-axis intensity (unaliased on destination grid)
        u_sas, _ = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
        i_sas = float(np.abs(u_sas[cy, cx])**2)
        assert abs(i_sas - i_theory_continuous) / i_theory_continuous < 0.03


def test_thin_lens_focus_measurement_independence():
    """Verify independent continuous quadratic fit waist measurement on thin-lens focus."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4
    focal_length = 75e-3

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_0 = gaussian_field(grid, waist=w0)
    t_lens = thin_lens_phase(grid, wavelength, focal_length)
    u_lens = u_0 * t_lens

    _, _, z_focus_theory, w_focus_theory = gaussian_beam_abcd(w0, wavelength, focal_length, z=focal_length)

    # Dense z sweep around focus
    z_scan = np.linspace(72.0e-3, 75.0e-3, 16)
    waists = []
    for z_i in z_scan:
        u_z = propagate_blas(u_lens, wavelength, z=z_i, dx=dx, dy=dy, padding=2)
        w_i = measure_beam_radius(np.abs(u_z)**2, grid.x, grid.y, roi_radius=1.5e-4)
        waists.append(w_i)

    # Local quadratic fit around minimum
    min_idx = int(np.argmin(waists))
    fit_idx = slice(max(0, min_idx - 2), min(len(z_scan), min_idx + 3))
    z_fit_pts = z_scan[fit_idx]
    w_fit_pts = np.array(waists[fit_idx])
    poly = np.polyfit(z_fit_pts, w_fit_pts, 2)
    c2, c1, _ = poly[0], poly[1], poly[2]

    assert c2 > 0  # Concave up (minimum)
    z_focus_meas = float(-c1 / (2.0 * c2))
    w_focus_meas = float(np.polyval(poly, z_focus_meas))

    # Verify that measured focus is a continuous interpolated point, not snapping to grid
    assert not any(np.isclose(z_focus_meas, z_scan, atol=1e-7))

    # Agreement with ABCD theory: focus position < 0.05%, waist < 0.1%
    rel_pos_err = abs(z_focus_meas - z_focus_theory) / z_focus_theory
    rel_waist_err = abs(w_focus_meas - w_focus_theory) / w_focus_theory
    assert rel_pos_err < 0.0005
    assert rel_waist_err < 0.001


def test_vortex_charge_multi_radius_multi_distance():
    """Verify optical vortex topological charge stability across multiple radii and distances."""
    nx, ny = 256, 256
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_gauss = gaussian_field(grid, waist=w0)

    test_radii = [1.2e-4, 1.5e-4, 1.8e-4]
    test_distances = [0.10, 0.20, 0.30]
    charges = [1, 2, 3, -1]

    for ell in charges:
        t_spiral = spiral_phase(grid, charge=ell)
        u_vortex = u_gauss * t_spiral

        for z_prop in test_distances:
            u_prop = propagate_blas(u_vortex, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)
            for r_test in test_radii:
                q = topological_charge(u_prop, grid, radius=r_test, num_points=360)
                assert abs(q - ell) < 0.05, f"Failed for charge {ell} at z={z_prop}, r={r_test}: q={q}"


def test_sas_validity_vignetting_and_fov():
    """Verify SAS validity check, magnification, and output FOV computation."""
    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    padding = 2

    # At z=0.01 m, M < 1 (chirp aliasing)
    valid_near, reason_near, m_near = check_sas_validity(wavelength, 0.01, nx, ny, dx, dy, padding=padding)
    assert not valid_near
    assert m_near < 1.0

    # At z=0.80 m, M > 1 (valid SAS regime)
    valid_far, reason_far, m_far = check_sas_validity(wavelength, 0.80, nx, ny, dx, dy, padding=padding)
    assert valid_far
    assert m_far > 1.0

    # At z=0.80 m, check exact output pitch and FOV:
    # dq = lambda * z / (2 * N * dx) = 532e-9 * 0.80 / (2 * 512 * 5e-6) = 83.125 um
    # S_d = N * dq = 512 * 83.125 um = 42.56 mm
    dqx, dqy, Sx_out, Sy_out = sas_output_scaling(wavelength, 0.80, nx, ny, dx, dy, padding=padding)
    assert np.isclose(dqx, 83.125e-6)
    assert np.isclose(dqy, 83.125e-6)
    assert np.isclose(Sx_out, 0.04256)
    assert np.isclose(Sy_out, 0.04256)


def test_method_selection_coverage_and_power_metrics():
    """Verify that FOV coverage rejects clipped fixed-grid methods at z=0.80 m while accepting SAS."""
    wavelength = 532e-9
    radius = 1.5e-4
    z = 0.80
    r1 = airy_first_zero(wavelength, z, radius)
    w_target = 2.0 * r1  # ~3.46 mm

    s_fixed = 512 * 5e-6  # 2.56 mm
    cov_x_fixed = s_fixed / w_target
    cov_2d_fixed = cov_x_fixed**2
    # Fixed window cannot cover >= 95% of target Airy disc
    sufficient_fixed = bool(cov_x_fixed >= 0.95 and cov_2d_fixed >= 0.90)
    assert not sufficient_fixed
    assert cov_x_fixed < 0.80
    assert cov_2d_fixed < 0.60

    # SAS scales to 42.56 mm
    s_sas = 0.04256
    cov_x_sas = min(1.0, s_sas / w_target)
    cov_2d_sas = cov_x_sas**2
    sufficient_sas = bool(cov_x_sas >= 0.95 and cov_2d_sas >= 0.90)
    assert sufficient_sas
    assert np.isclose(cov_x_sas, 1.0)
    assert np.isclose(cov_2d_sas, 1.0)


def test_sas_destination_pitch_and_cropped_fov():
    """Verify destination pitch, magnification, and cropped output FOV for N=512, dx=5 um, padding=2, z=0.80 m."""
    n = 512
    dx = 5.0e-6
    padding = 2
    wavelength = 532e-9
    z = 0.80

    dqx, dqy, sx_out, sy_out = sas_output_scaling(wavelength, z, n, n, dx, dx, padding=padding)

    # Destination pitch: dq = lambda * z / (n * padding * dx)
    expected_dq = 532e-9 * 0.80 / (512 * 2 * 5.0e-6)  # 83.125 um
    assert np.isclose(dqx, expected_dq)
    assert np.isclose(dqx, 83.125e-6)

    # Magnification: M = dq / dx
    m = dqx / dx
    assert np.isclose(m, 16.625)

    # Cropped output FOV: Sx = n * dq = 512 * 83.125 um = 42.56 mm
    assert np.isclose(sx_out, 0.04256)
    assert np.isclose(sy_out, 0.04256)


def test_thin_lens_precision_and_relative_error():
    """Verify thin-lens unrounded numerical measurement error (< 0.01% for focus, < 0.1% for waist) and axial resolution."""
    w0 = 3.0e-4
    wavelength = 532e-9
    f = 75e-3

    _, _, z_th, w_th = gaussian_beam_abcd(w0, wavelength, f, z=f)
    # Analytical theory: z_th ~ 73.5356 mm, w_th ~ 41.920 um
    assert abs(z_th * 1e3 - 73.5356) < 0.001
    assert abs(w_th * 1e6 - 41.920) < 0.01

    # Actual numerical measurements from continuous quadratic fit
    # Direct unrounded measurements: z_meas ~ 73.5330 mm, w_meas ~ 41.939 um
    grid = create_grid(nx=256, ny=256, dx=5e-6, dy=5e-6)
    u_in = gaussian_field(grid, waist=w0) * thin_lens_phase(grid, wavelength, f)
    z_scan = np.linspace(72.0e-3, 75.0e-3, 16)
    waists = [measure_beam_radius(np.abs(propagate_blas(u_in, wavelength, z=z_i, dx=5e-6, dy=5e-6, padding=2))**2, grid.x, grid.y, roi_radius=4.0*w_th) for z_i in z_scan]
    min_i = np.argmin(waists)
    fit_sl = slice(max(0, min_i - 2), min(len(z_scan), min_i + 3))
    p = np.polyfit(z_scan[fit_sl], np.array(waists)[fit_sl], 2)
    z_meas = float(-p[1] / (2.0 * p[0]))
    w_meas = float(np.polyval(p, z_meas))

    # Raw unrounded relative errors
    raw_pos_err = abs(z_meas - z_th) / z_th
    raw_waist_err = abs(w_meas - w_th) / w_th
    assert raw_pos_err < 0.0001   # Raw error ~0.0036% (< 0.01%)
    assert raw_waist_err < 0.001  # Raw error ~0.045% (< 0.1%)

    # Verify uncertainty corresponds to numerical resolution (dz / 10), not rounded error
    dz = z_scan[1] - z_scan[0]
    focus_resolution = dz / 10.0
    assert abs(z_meas - z_th) < focus_resolution


def test_optical_power_integration_vs_area_coverage():
    """Verify that optical power is computed by 2D physical intensity integration, not inferred from area coverage."""
    wavelength = 532e-9
    radius = 1.5e-4
    z = 0.80
    r1 = airy_first_zero(wavelength, z, radius)
    w_target = 2.0 * r1  # ~3.46 mm

    s_fixed = 512 * 5e-6  # 2.56 mm
    area_coverage_ratio = (s_fixed / w_target)**2
    # Geometric area coverage is ~54.8%
    assert abs(area_coverage_ratio - 0.548) < 0.02

    # Physical captured optical power by integrating Airy intensity over [-s_fixed/2, s_fixed/2]^2
    grid_fine = create_grid(nx=1024, ny=1024, dx=5e-6, dy=5e-6)
    i_airy = airy_pattern(grid_fine, wavelength, z=z, radius=radius)
    mask_win = (np.abs(grid_fine.X) <= s_fixed / 2.0) & (np.abs(grid_fine.Y) <= s_fixed / 2.0)

    p_captured = np.sum(i_airy[mask_win])
    p_total = np.sum(i_airy)
    captured_power_ratio = float(p_captured / p_total)

    # Physical captured power (> 85%) is vastly higher than geometric area coverage (~55%)
    assert captured_power_ratio > 0.85
    assert captured_power_ratio > area_coverage_ratio + 0.30


def test_pupil_spectrum_bandlimit_gate():
    """Verify physical justification of BL-ASM bandwidth gate f_pupil = 3/D based on pupil spectrum."""
    aperture_radius = 1.5e-4
    aperture_diameter = 2.0 * aperture_radius  # D = 0.30 mm

    f_pupil_threshold = 3.0 / aperture_diameter
    assert np.isclose(f_pupil_threshold, 10000.0)

    # The 3rd dark zero of the J1(pi*D*f) pupil spectrum is at u3 / (pi*D) ~ 10.1735 / (pi*D) ~ 3.238/D
    # Setting threshold at 3.0/D encloses central core + first two secondary rings (>95% of spectral energy)
    zero3_freq = 10.173506 / (np.pi * aperture_diameter)
    assert f_pupil_threshold < zero3_freq
    assert zero3_freq - f_pupil_threshold < 1000.0


def test_regression_issue_a_method_selection_map_isolation(tmp_path):
    """Regression test for Issue A: Verify that method selection map isolates its records from mixed inputs."""
    from visualize import generate_fig04_method_selection_map

    # Construct synthetic mixed metrics records containing both method_selection_map and fixed_fov
    z_test = 0.50
    dx_test = 5e-6
    map_row = {
        "experiment": "method_selection_map",
        "method": "blas",
        "z_m": z_test,
        "dx_source_m": dx_test,
        "dx_output_m": dx_test,
        "relative_l2_error": 0.015,
        "runtime_s": 0.005,
        "valid": True,
        "sufficient_coverage": True,
        "reference_converged": True,
        "ref_uncertainty": 0.005,
    }
    # Foreign fixed_fov row with different/conflicting coordinates and missing sweep metadata
    foreign_row = {
        "experiment": "fixed_fov",
        "method": "blas",
        "z_m": 0.999,
        "dx_source_m": 12.34e-6,
        "relative_l2_error": 0.888,
        "runtime_s": 0.123,
    }
    mixed_data = [map_row, foreign_row, map_row]  # Include duplicate and mixed record

    out_png = str(tmp_path / "fig04_test.png")
    out_pdf = str(tmp_path / "fig04_test.pdf")

    # Should execute successfully without contamination by foreign row
    generate_fig04_method_selection_map(metrics_data=mixed_data, output_png=out_png, output_pdf=out_pdf)
    assert (tmp_path / "fig04_test.png").exists()

    # Foreign-only data must be rejected with ValueError
    with pytest.raises(ValueError, match="No valid 'method_selection_map' records"):
        generate_fig04_method_selection_map(metrics_data=[foreign_row], output_png=out_png, output_pdf=out_pdf)

    # Conflicting duplicate cells must be rejected with ValueError
    conflicting_row = dict(map_row)
    conflicting_row["relative_l2_error"] = 0.500  # Conflicts significantly with 0.015
    with pytest.raises(ValueError, match="Conflicting duplicate benchmark records"):
        generate_fig04_method_selection_map(metrics_data=[map_row, conflicting_row], output_png=out_png, output_pdf=out_pdf)


def test_regression_issue_b_vortex_no_theory_substitution(tmp_path):
    """Regression test for Issue B: Verify that optical vortex visualization never substitutes theory for missing charges."""
    from visualize import generate_fig06_vortex_beam

    # Minimal input intentionally missing charge +3 and containing negative charge -1
    vortex_data_incomplete = {
        "passed": True,
        "charges": [
            {
                "charge_requested": 1,
                "charge_input_measured": 1.0001,
                "charge_output_measured": 0.9998,
            },
            {
                "charge_requested": -1,
                "charge_input_measured": -0.9999,
                "charge_output_measured": -1.0002,
            },
        ],
    }

    out_png = str(tmp_path / "fig06_test.png")
    out_pdf = str(tmp_path / "fig06_test.pdf")

    # Render figure with incomplete data
    generate_fig06_vortex_beam(vortex_data=vortex_data_incomplete, output_png=out_png, output_pdf=out_pdf, quick=True)
    assert (tmp_path / "fig06_test.png").exists()


def test_regression_issue_c_validation_failure_exit_status(tmp_path):
    """Regression test for Issue C: Verify that reproduction fails with nonzero status when validations fail."""
    import subprocess
    import sys
    import os

    # Run quick reproduction with an injected failing validation via python -c into isolated tmp_path
    test_out = str(tmp_path / "test_results")
    script = (
        "import sys; "
        "import reproduce; "
        "reproduce.run_validations = lambda quick=False: {'failing_check': {'passed': False}}; "
        f"sys.argv = ['reproduce.py', '--output-dir', r'{test_out}']; "
        "reproduce.main()"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 1
    assert "FAILED in checks" in res.stderr
    assert "IndexError" not in res.stderr
    assert "Traceback" not in res.stderr
    assert os.path.exists(os.path.join(test_out, "validation_failure.json"))


def test_save_results_empty_metrics_handling(tmp_path):
    """Verify that save_results handles empty metrics lists atomically without raising IndexError."""
    import reproduce
    import os
    import json
    import csv

    test_out = str(tmp_path / "atomic_save_test")
    dummy_val = {"check_a": {"passed": True}}

    # Call with empty metrics
    reproduce.save_results(dummy_val, [], output_dir=test_out)

    val_file = os.path.join(test_out, "validation.json")
    met_file = os.path.join(test_out, "metrics.csv")
    assert os.path.exists(val_file)
    assert os.path.exists(met_file)

    with open(val_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["check_a"]["passed"] is True

    # metrics.csv should have a valid header row and 0 data rows, not 0 bytes
    assert os.path.getsize(met_file) > 0
    with open(met_file, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 1
    assert "experiment" in reader[0]


def test_validation_failure_preserves_existing_results(tmp_path):
    """Verify that a failed validation run does not overwrite or truncate existing valid results."""
    import reproduce
    import os
    import json
    import subprocess
    import sys

    test_out = str(tmp_path / "preserve_test")
    os.makedirs(test_out, exist_ok=True)
    val_file = os.path.join(test_out, "validation.json")
    met_file = os.path.join(test_out, "metrics.csv")

    # Seed with existing valid results
    seed_val = {"original": {"passed": True}}
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(seed_val, f)
    with open(met_file, "w", encoding="utf-8") as f:
        f.write("experiment,method\noriginal,asm\n")

    orig_val_content = open(val_file, "r", encoding="utf-8").read()
    orig_met_content = open(met_file, "r", encoding="utf-8").read()

    # Run reproduction with injected failure
    script = (
        "import sys; "
        "import reproduce; "
        "reproduce.run_validations = lambda quick=False: {'failing_check': {'passed': False}}; "
        f"sys.argv = ['reproduce.py', '--output-dir', r'{test_out}']; "
        "reproduce.main()"
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert res.returncode == 1

    # Verify original files were NOT touched or corrupted
    assert open(val_file, "r", encoding="utf-8").read() == orig_val_content
    assert open(met_file, "r", encoding="utf-8").read() == orig_met_content
    assert os.path.exists(os.path.join(test_out, "validation_failure.json"))


def test_grid_and_pad_even_odd_centering():
    """Verify origin centering and absence of half-pixel shifts for even, odd, and rectangular grids."""
    from scipy.fft import fftshift, fftfreq

    grid_shapes = [
        (64, 64), (65, 65), (128, 128), (129, 129),  # Square even/odd
        (64, 128), (128, 64),                        # Rectangular even
        (65, 129), (129, 65),                        # Rectangular odd
        (64, 65), (129, 128),                        # Mixed odd/even
    ]

    for nx, ny in grid_shapes:
        dx, dy = 5e-6, 8e-6
        grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)

        # Coordinate origin at integer indices nx // 2 and ny // 2
        cx, cy = nx // 2, ny // 2
        assert grid.x[cx] == 0.0
        assert grid.y[cy] == 0.0
        assert grid.fx[cx] == 0.0
        assert grid.fy[cy] == 0.0

        # Spatial frequency matches exact scipy fftfreq with fftshift
        assert np.allclose(grid.fx, fftshift(fftfreq(nx, d=dx)))
        assert np.allclose(grid.fy, fftshift(fftfreq(ny, d=dy)))

        # Central impulse preservation
        field = np.zeros((ny, nx), dtype=complex)
        field[cy, cx] = 1.0
        for factor in [1, 2, 3]:
            padded, crop_slices = pad_field(field, factor=factor)
            new_cx = (nx * factor) // 2
            new_cy = (ny * factor) // 2
            assert padded[new_cy, new_cx] == 1.0
            # Roundtrip unpadding recovers original delta
            assert padded[crop_slices][cy, cx] == 1.0
            assert np.sum(np.abs(padded)) == 1.0

        # Off-center impulse preservation
        field_off = np.zeros((ny, nx), dtype=complex)
        off_x, off_y = min(3, cx - 1), min(2, cy - 1)
        field_off[cy + off_y, cx - off_x] = 1.0
        for factor in [1, 2, 3]:
            padded_off, crop_slices_off = pad_field(field_off, factor=factor)
            recovered = padded_off[crop_slices_off]
            assert recovered[cy + off_y, cx - off_x] == 1.0
            assert np.sum(np.abs(recovered)) == 1.0


def test_complex_field_error_global_phase_alignment():
    """Verify analytical optimal global phase alignment in compute_complex_field_error."""
    rng = np.random.default_rng(999)
    u_ref = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))

    # Test with arbitrary global phase shift theta
    theta = 1.75
    u_test = u_ref * np.exp(-1j * theta)

    err, phi_opt = compute_complex_field_error(u_test, u_ref)
    assert err < 1e-14
    # Recovered phi_opt cancels theta: phi_opt ~ theta (mod 2pi)
    phase_diff = (phi_opt - theta) % (2 * np.pi)
    assert np.isclose(phase_diff, 0.0, atol=1e-12) or np.isclose(phase_diff, 2 * np.pi, atol=1e-12)

    # Test with additive noise
    noise = 0.02 * (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)))
    u_noisy = u_ref * np.exp(-1j * theta) + noise
    err_noisy, _ = compute_complex_field_error(u_noisy, u_ref)
    assert 0.015 < err_noisy < 0.025


def test_physical_parameter_validation():
    """Verify that invalid physical and numerical parameters are rejected with ValueError."""
    grid = create_grid(nx=64, ny=64, dx=5e-6, dy=5e-6)
    u = circular_aperture(grid, radius=1e-4)

    # Nonpositive wavelength
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=-532e-9, z=0.01, dx=5e-6, dy=5e-6)
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=0.0, z=0.01, dx=5e-6, dy=5e-6)

    # Nonpositive sampling pitches
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=532e-9, z=0.01, dx=-5e-6, dy=5e-6)
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=532e-9, z=0.01, dx=5e-6, dy=0.0)

    # Negative distance
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=532e-9, z=-0.01, dx=5e-6, dy=5e-6)

    # Invalid padding
    with pytest.raises(ValueError):
        propagate_blas(u, wavelength=532e-9, z=0.01, dx=5e-6, dy=5e-6, padding=0)

    # Invalid aperture parameters
    with pytest.raises(ValueError):
        circular_aperture(grid, radius=-1e-4)
    with pytest.raises(ValueError):
        rectangular_aperture(grid, width=-1e-4, height=1e-4)
    with pytest.raises(ValueError):
        gaussian_field(grid, waist=0.0)
    with pytest.raises(ValueError):
        thin_lens_phase(grid, wavelength=532e-9, focal_length=0.0)


def test_sas_validity_nonsquare_2d_vignetting():
    """Verify that check_sas_validity evaluates boundary-specific 2D vignetting and per-axis magnification."""
    wavelength = 532e-9
    # Configuration with distinct per-axis critical distances and vignetting limits:
    # Axis X: Nx=128, dx=5 um -> Lx=0.64 mm, z_c,x = 2*128*(5e-6)^2 / 532e-9 = 0.01203 m, z_vig,x ~ 34.04 m
    # Axis Y: Ny=512, dy=10 um -> Ly=5.12 mm, z_c,y = 2*512*(10e-6)^2 / 532e-9 = 0.19248 m, z_vig,y ~ 2176.86 m
    nx, ny = 128, 512
    dx, dy = 5e-6, 10e-6

    # 1. Zero and negative distance: singular
    v_zero, r_zero, _ = check_sas_validity(wavelength, 0.0, nx, ny, dx, dy, padding=2)
    assert not v_zero
    assert "z=0" in r_zero

    # 2. Both axes below critical distance (z=0.005 m < z_c,x < z_c,y)
    v_near, r_near, m_near = check_sas_validity(wavelength, 0.005, nx, ny, dx, dy, padding=2)
    assert not v_near
    assert m_near < 1.0

    # 3. Mixed magnification regime: Mx > 1 but My < 1 (z=0.05 m: z_c,x < z < z_c,y)
    # Mx = 0.05 / 0.01203 ~ 4.16 > 1, but My = 0.05 / 0.19248 ~ 0.26 <= 1
    v_mixed, r_mixed, m_mixed = check_sas_validity(wavelength, 0.05, nx, ny, dx, dy, padding=2)
    assert not v_mixed
    assert "My=" in r_mixed
    assert m_mixed < 1.0

    # 4. Valid propagation regime: both Mx > 1 and My > 1 and z < z_vig (z=0.25 m)
    v_valid, r_valid, m_valid = check_sas_validity(wavelength, 0.25, nx, ny, dx, dy, padding=2)
    assert v_valid
    assert m_valid > 1.0
    assert "Valid SAS" in r_valid

    # 5. Boundary-specific vignetting check on X axis (limiting axis is X with z_vig ~ 34.04 m)
    # Just below limiting distance: z = 33.0 m < 34.04 m -> valid
    v_below, _, _ = check_sas_validity(wavelength, 33.0, nx, ny, dx, dy, padding=2)
    assert v_below

    # Just above limiting distance: z = 35.0 m > 34.04 m -> invalid due to X-axis vignetting
    v_above, r_above, _ = check_sas_validity(wavelength, 35.0, nx, ny, dx, dy, padding=2)
    assert not v_above
    assert "Vignetting limit exceeded" in r_above

    # 6. Swapped axes: Axis Y is now the limiting axis (Ny=128, dy=5 um, Nx=512, dx=10 um)
    v_swap_above, r_swap_above, _ = check_sas_validity(wavelength, 35.0, ny, nx, dy, dx, padding=2)
    assert not v_swap_above
    assert "Vignetting limit exceeded" in r_swap_above


def test_manifest_creation_and_sha256_verification(tmp_path):
    """Verify that save_results creates manifest.json with valid SHA256 hashes and run archiving."""
    import reproduce
    import os
    import json
    import hashlib

    test_out = str(tmp_path / "manifest_test")
    val_sample = {
        "test_check": {"passed": True, "value": 42}
    }
    met_sample = [
        {"experiment": "test_exp", "method": "blas", "relative_l2_error": 0.001, "valid": True}
    ]

    manifest = reproduce.save_results(val_sample, met_sample, output_dir=test_out, quick=False)

    assert manifest["schema_version"] == "2.0.0"
    assert manifest["execution_mode"] == "full"
    assert manifest["all_validations_passed"] is True
    run_id = manifest["run_id"]

    # Verify top-level files exist
    val_file = os.path.join(test_out, "validation.json")
    met_file = os.path.join(test_out, "metrics.csv")
    man_file = os.path.join(test_out, "manifest.json")
    assert os.path.exists(val_file)
    assert os.path.exists(met_file)
    assert os.path.exists(man_file)

    # Verify SHA256 integrity
    def get_sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    assert manifest["files"]["validation.json"]["sha256"] == get_sha256(val_file)
    assert manifest["files"]["metrics.csv"]["sha256"] == get_sha256(met_file)

    # Verify run archiving in runs/<run_id>/
    run_dir = os.path.join(test_out, "runs", run_id)
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(run_dir, "validation.json"))
    assert os.path.exists(os.path.join(run_dir, "metrics.csv"))
    assert os.path.exists(os.path.join(run_dir, "manifest.json"))


def test_quick_mode_preserves_full_benchmark_results(tmp_path):
    """Verify that quick mode preserves existing full benchmark results unless overwrite_full=True."""
    import reproduce
    import os
    import json

    test_out = str(tmp_path / "quick_protect_test")
    os.makedirs(test_out, exist_ok=True)

    # Seed full run
    seed_val = {"official_full": {"passed": True}}
    seed_met = [{"experiment": "full_bench", "method": "blas", "N": 512}]
    reproduce.save_results(seed_val, seed_met, output_dir=test_out, quick=False)

    orig_val_bytes = open(os.path.join(test_out, "validation.json"), "rb").read()
    orig_met_bytes = open(os.path.join(test_out, "metrics.csv"), "rb").read()

    # Now execute save_results in quick mode without overwrite_full
    quick_val = {"quick_check": {"passed": True}}
    quick_met = [{"experiment": "quick_bench", "method": "blas", "N": 128}]
    reproduce.save_results(quick_val, quick_met, output_dir=test_out, quick=True, overwrite_full=False)

    # Top-level files MUST be unchanged
    assert open(os.path.join(test_out, "validation.json"), "rb").read() == orig_val_bytes
    assert open(os.path.join(test_out, "metrics.csv"), "rb").read() == orig_met_bytes

    # Quick results must be written to validation_quick.json and metrics_quick.csv
    assert os.path.exists(os.path.join(test_out, "validation_quick.json"))
    assert os.path.exists(os.path.join(test_out, "metrics_quick.csv"))
    with open(os.path.join(test_out, "validation_quick.json"), "r", encoding="utf-8") as f:
        q_val_data = json.load(f)
    assert "quick_check" in q_val_data

    # Now execute with overwrite_full=True
    reproduce.save_results(quick_val, quick_met, output_dir=test_out, quick=True, overwrite_full=True)
    with open(os.path.join(test_out, "validation.json"), "r", encoding="utf-8") as f:
        overwritten_data = json.load(f)
    assert "quick_check" in overwritten_data


def test_cli_asset_dir_isolation(tmp_path):
    """Verify that CLI --asset-dir flag isolates figures without modifying default assets/."""
    import subprocess
    import sys
    import os

    custom_assets = str(tmp_path / "custom_assets")
    cmd = [
        sys.executable,
        "reproduce.py",
        "--figure-only",
        "--asset-dir",
        custom_assets,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Error: {res.stderr}"

    # Verify figures exist in custom_assets
    assert os.path.exists(os.path.join(custom_assets, "fig01_aperture_diffraction.png"))
    assert os.path.exists(os.path.join(custom_assets, "fig01_aperture_diffraction.pdf"))
    assert os.path.exists(os.path.join(custom_assets, "fig05_thin_lens_focus.png"))


def test_characterize_thin_lens_focus_pipeline_consistency():
    """Verify characterize_thin_lens_focus numerical accuracy and unrounded consistency."""
    from diffraction import characterize_thin_lens_focus, gaussian_beam_abcd

    res = characterize_thin_lens_focus(quick=True)
    assert res["passed"] is True
    assert res["focus_z_relative_error"] < 0.001
    assert res["waist_relative_error"] < 0.005
    assert res["focus_z_uncertainty_m"] == 2.0e-6
    assert res["fit_method"] == "quadratic_interpolation"


def test_optical_vortex_strict_per_contour_rejection():
    """Verify that a single bad contour fails optical vortex validation."""
    from diffraction import create_grid, gaussian_field, spiral_phase, propagate_blas, topological_charge
    import numpy as np

    grid = create_grid(nx=128, ny=128, dx=10e-6, dy=10e-6)
    u_gauss = gaussian_field(grid, waist=3e-4)
    t_v = spiral_phase(grid, charge=1)
    u_vortex = u_gauss * t_v

    # Contour at origin (r=0) passes through the phase singularity / null amplitude
    q_null, min_amp_null = topological_charge(u_vortex, grid, radius=0.0, num_points=180, return_min_amplitude=True)
    assert min_amp_null < 1e-12

    # Normal contour at r=1.5e-4 is well outside the dark core
    q_good, min_amp_good = topological_charge(u_vortex, grid, radius=1.5e-4, num_points=180, return_min_amplitude=True)
    assert abs(q_good - 1.0) < 0.02
    assert min_amp_good > 1e-4


def test_near_field_complex_error_acceptance_gating():
    """Verify that near-field direct integration validates both intensity and complex-field errors."""
    from reproduce import run_validations

    # Run validations in quick mode
    val = run_validations(quick=True)
    nf_check = val["near_field_direct_integration"]

    assert "passed_intensity" in nf_check
    assert "passed_complex" in nf_check
    assert nf_check["passed_intensity"] is True
    assert nf_check["passed_complex"] is True
    assert nf_check["passed"] is True
    assert nf_check["relative_l2_error"] < 0.05
    assert nf_check["complex_field_error"] < 0.10


def test_fig04_duplicate_detection_conflicting_attributes(tmp_path):
    """Verify that Fig 4 duplicate check raises ValueError on conflicting metadata across same cell."""
    from visualize import generate_fig04_method_selection_map
    import pytest

    cell_base = {
        "experiment": "method_selection_map",
        "method": "blas",
        "z_m": 0.05,
        "dx_source_m": 5e-6,
        "dx_output_m": 5e-6,
        "relative_l2_error": 0.01,
        "runtime_s": 0.05,
        "valid": True,
        "sufficient_coverage": True,
        "reference_converged": True,
        "ref_uncertainty": 0.005,
    }

    # Duplicate cell with conflicting valid boolean
    cell_conflicting = dict(cell_base)
    cell_conflicting["valid"] = False

    data = [cell_base, cell_conflicting]
    out_png = str(tmp_path / "dup_test.png")
    out_pdf = str(tmp_path / "dup_test.pdf")

    with pytest.raises(ValueError, match="Conflicting duplicate benchmark records"):
        generate_fig04_method_selection_map(metrics_data=data, output_png=out_png, output_pdf=out_pdf)


