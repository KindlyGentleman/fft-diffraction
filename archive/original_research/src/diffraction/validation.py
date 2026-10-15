from typing import Dict, Any, Tuple
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture, make_rectangular_aperture
from diffraction.propagation import (
    propagate_asm,
    propagate_blas,
    propagate_fresnel,
    propagate_direct_integration,
)
from diffraction.analytical import (
    airy_first_zero_radius,
    circular_fraunhofer_intensity,
    rectangular_fraunhofer_intensity,
)
from diffraction.metrics import (
    calculate_intensity,
    calculate_energy,
    relative_l2_error,
    find_first_radial_minimum,
)
from diffraction.utils import radial_average

def validate_zero_distance(
    nx: int = 256,
    ny: int = 256,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
) -> Dict[str, Any]:
    """Test A: Verify that all methods at z=0 reproduce the input field to numerical precision."""
    grid = create_grid(nx, ny, dx, dy)
    field_circ = make_circular_aperture(grid, radius=1.5e-4)

    res_asm, _, _ = propagate_asm(field_circ, wavelength, z=0.0, dx=dx, dy=dy, padding=2)
    res_blas, _, _ = propagate_blas(field_circ, wavelength, z=0.0, dx=dx, dy=dy, padding=2)
    res_fresnel, _, _ = propagate_fresnel(field_circ, wavelength, z=0.0, dx=dx, dy=dy, padding=2)

    err_asm = relative_l2_error(res_asm, field_circ)
    err_blas = relative_l2_error(res_blas, field_circ)
    err_fresnel = relative_l2_error(res_fresnel, field_circ)

    passed = (err_asm < 1e-12) and (err_blas < 1e-12) and (err_fresnel < 1e-12)
    return {
        "test": "TEST_A_ZERO_DISTANCE",
        "passed": passed,
        "error_asm": float(err_asm),
        "error_blas": float(err_blas),
        "error_fresnel": float(err_fresnel),
    }

def validate_plane_wave(
    nx: int = 256,
    ny: int = 256,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    z: float = 0.05,
) -> Dict[str, Any]:
    """Test B: Verify uniform plane wave phase evolution exp(i*k*z) and constant intensity."""
    field_pw = np.ones((ny, nx), dtype=np.complex128)
    k = 2.0 * np.pi / wavelength
    expected_field = np.exp(1j * k * z) * field_pw

    res_asm, _, _ = propagate_asm(field_pw, wavelength, z=z, dx=dx, dy=dy, padding=1)
    res_blas, _, _ = propagate_blas(field_pw, wavelength, z=z, dx=dx, dy=dy, padding=1)
    res_fresnel, _, _ = propagate_fresnel(field_pw, wavelength, z=z, dx=dx, dy=dy, padding=1)

    err_asm = relative_l2_error(res_asm, expected_field)
    err_blas = relative_l2_error(res_blas, expected_field)
    err_fresnel = relative_l2_error(res_fresnel, expected_field)

    passed = (err_asm < 1e-10) and (err_blas < 1e-10) and (err_fresnel < 1e-10)
    return {
        "test": "TEST_B_PLANE_WAVE",
        "passed": passed,
        "error_asm": float(err_asm),
        "error_blas": float(err_blas),
        "error_fresnel": float(err_fresnel),
    }

def validate_fft_consistency(nx: int = 256, ny: int = 256) -> Dict[str, Any]:
    """Test C: Verify FFT/IFFT roundtrip invertibility and coordinate shift consistency."""
    rng = np.random.default_rng(42)
    test_field = rng.standard_normal((ny, nx)) + 1j * rng.standard_normal((ny, nx))

    roundtrip = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(test_field))))))
    err = float(np.linalg.norm(roundtrip - test_field) / np.linalg.norm(test_field))

    passed = err < 1e-14
    return {
        "test": "TEST_C_FFT_CONSISTENCY",
        "passed": passed,
        "roundtrip_error": err,
    }

def validate_circular_fraunhofer(
    nx: int = 512,
    ny: int = 512,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    radius: float = 1.5e-4,
    z: float = 0.500,
) -> Dict[str, Any]:
    """Test D: Compare numerical Fraunhofer pattern with analytical Airy disc solution."""
    grid = create_grid(nx, ny, dx, dy)
    field_in = make_circular_aperture(grid, radius=radius)

    u_blas, _, _ = propagate_blas(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
    i_num = calculate_intensity(u_blas)
    i_ana = circular_fraunhofer_intensity(grid, wavelength, z=z, radius=radius)

    r_num, prof_num = radial_average(i_num, grid.x, grid.y, num_bins=150)
    r_ana, prof_ana = radial_average(i_ana, grid.x, grid.y, num_bins=150)

    r1_theory = airy_first_zero_radius(wavelength, z, radius)
    r1_meas = find_first_radial_minimum(r_num, prof_num, search_range=(0.7 * r1_theory, 1.4 * r1_theory))
    r1_err_rel = abs(r1_meas - r1_theory) / r1_theory

    norm_prof_num = prof_num / np.max(prof_num) if np.max(prof_num) > 0 else prof_num
    norm_prof_ana = prof_ana / np.max(prof_ana) if np.max(prof_ana) > 0 else prof_ana
    prof_l2_err = float(np.linalg.norm(norm_prof_num - norm_prof_ana) / np.linalg.norm(norm_prof_ana))

    passed = (r1_err_rel < 0.08) and (prof_l2_err < 0.12)
    return {
        "test": "TEST_D_CIRCULAR_FRAUNHOFER",
        "passed": passed,
        "r1_theory_m": float(r1_theory),
        "r1_measured_m": float(r1_meas),
        "r1_relative_error": float(r1_err_rel),
        "profile_l2_error": float(prof_l2_err),
    }

def validate_rectangular_fraunhofer(
    nx: int = 512,
    ny: int = 512,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    width: float = 3.0e-4,
    height: float = 4.5e-4,
    z: float = 0.500,
) -> Dict[str, Any]:
    """Test E: Compare numerical Fraunhofer pattern with analytical sinc^2 cross sections."""
    grid = create_grid(nx, ny, dx, dy)
    field_in = make_rectangular_aperture(grid, width=width, height=height)

    u_blas, _, _ = propagate_blas(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
    i_num = calculate_intensity(u_blas)
    i_ana = rectangular_fraunhofer_intensity(grid, wavelength, z=z, width=width, height=height)

    cy, cx = ny // 2, nx // 2
    x_slice_num = i_num[cy, :] / np.max(i_num[cy, :])
    x_slice_ana = i_ana[cy, :] / np.max(i_ana[cy, :])
    y_slice_num = i_num[:, cx] / np.max(i_num[:, cx])
    y_slice_ana = i_ana[:, cx] / np.max(i_ana[:, cx])

    err_x = float(np.linalg.norm(x_slice_num - x_slice_ana) / np.linalg.norm(x_slice_ana))
    err_y = float(np.linalg.norm(y_slice_num - y_slice_ana) / np.linalg.norm(y_slice_ana))

    passed = (err_x < 0.12) and (err_y < 0.12)
    return {
        "test": "TEST_E_RECTANGULAR_FRAUNHOFER",
        "passed": passed,
        "slice_x_error": float(err_x),
        "slice_y_error": float(err_y),
    }

def validate_near_field_direct_integration(
    nx: int = 64,
    ny: int = 64,
    dx: float = 8.0e-6,
    dy: float = 8.0e-6,
    wavelength: float = 5.32e-7,
    radius: float = 8.0e-5,
    z: float = 0.005,
) -> Dict[str, Any]:
    """Test F: Cross-check BL-ASM against non-FFT direct Rayleigh-Sommerfeld integration."""
    grid = create_grid(nx, ny, dx, dy)
    field_in = make_circular_aperture(grid, radius=radius)

    u_blas, _, _ = propagate_blas(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    eval_indices_x = np.linspace(nx // 4, 3 * nx // 4, 16, dtype=int)
    eval_indices_y = np.linspace(ny // 4, 3 * ny // 4, 16, dtype=int)
    sub_x = grid.x[eval_indices_x]
    sub_y = grid.y[eval_indices_y]

    u_rs = propagate_direct_integration(
        field=field_in,
        wavelength=wavelength,
        z=z,
        dx=dx,
        dy=dy,
        x_out=sub_x,
        y_out=sub_y,
    )

    u_blas_sub = u_blas[np.ix_(eval_indices_y, eval_indices_x)]
    i_blas_sub = calculate_intensity(u_blas_sub)
    i_rs = calculate_intensity(u_rs)

    err_l2 = float(np.linalg.norm(i_blas_sub - i_rs) / np.linalg.norm(i_rs))
    passed = err_l2 < 0.08
    return {
        "test": "TEST_F_NEAR_FIELD_DIRECT_INTEGRATION",
        "passed": passed,
        "relative_l2_error": float(err_l2),
    }

def validate_energy_conservation(
    nx: int = 256,
    ny: int = 256,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    z: float = 0.010,
) -> Dict[str, Any]:
    """Test H: Audit energy conservation across propagation methods."""
    grid = create_grid(nx, ny, dx, dy)
    field_in = make_circular_aperture(grid, radius=1.0e-4)

    e_in = calculate_energy(field_in, dx, dy)

    u_asm, _, _ = propagate_asm(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
    u_blas, _, _ = propagate_blas(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
    u_fresnel, _, _ = propagate_fresnel(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    e_asm = calculate_energy(u_asm, dx, dy)
    e_blas = calculate_energy(u_blas, dx, dy)
    e_fresnel = calculate_energy(u_fresnel, dx, dy)

    disc_asm = abs(e_asm - e_in) / e_in
    disc_blas = abs(e_blas - e_in) / e_in
    disc_fresnel = abs(e_fresnel - e_in) / e_in

    passed = (disc_asm < 0.02) and (disc_blas < 0.05) and (disc_fresnel < 0.02)
    return {
        "test": "TEST_H_ENERGY_CONSERVATION",
        "passed": passed,
        "input_energy": float(e_in),
        "discrepancy_asm": float(disc_asm),
        "discrepancy_blas": float(disc_blas),
        "discrepancy_fresnel": float(disc_fresnel),
    }

def validate_waveprop_comparison(
    nx: int = 256,
    ny: int = 256,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    radius: float = 1.5e-4,
    z: float = 0.050,
) -> Dict[str, Any]:
    """Test I: Cross-check results against reference package waveprop."""
    try:
        import waveprop.rs as wp_rs
        from waveprop.util import circ
    except ImportError:
        return {
            "test": "TEST_I_WAVEPROP_COMPARISON",
            "passed": False,
            "status": "LIMITATION_NOT_AVAILABLE",
            "error": "waveprop library not available",
        }

    grid = create_grid(nx, ny, dx, dy)
    field_in = make_circular_aperture(grid, radius=radius)

    u_our_blas, _, _ = propagate_blas(field_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

    wp_in = circ(grid.X, grid.Y, diam=2.0 * radius).astype(np.complex128)
    u_wp, _, _ = wp_rs.angular_spectrum(
        u_in=wp_in,
        wv=wavelength,
        d1=[dy, dx],
        dz=z,
        bandlimit=True,
        pad=True,
    )

    i_our = calculate_intensity(u_our_blas)
    i_wp = calculate_intensity(u_wp)

    err_l2 = float(np.linalg.norm(i_our - i_wp) / np.linalg.norm(i_wp))
    passed = err_l2 < 0.01

    return {
        "test": "TEST_I_WAVEPROP_COMPARISON",
        "passed": passed,
        "intensity_relative_l2_error": float(err_l2),
    }

def run_full_validation_suite() -> Dict[str, Any]:
    """Execute all validation tests and compile structured results."""
    results = {}
    results["TEST_A"] = validate_zero_distance()
    results["TEST_B"] = validate_plane_wave()
    results["TEST_C"] = validate_fft_consistency()
    results["TEST_D"] = validate_circular_fraunhofer()
    results["TEST_E"] = validate_rectangular_fraunhofer()
    results["TEST_F"] = validate_near_field_direct_integration()
    results["TEST_H"] = validate_energy_conservation()
    results["TEST_I"] = validate_waveprop_comparison()

    all_passed = all(res.get("passed", False) for res in results.values())
    results["ALL_TESTS_PASSED"] = all_passed
    return results
