"""Reproduction CLI runner for FFT scalar diffraction experiments.

Runs validation tests, generates metrics tables, and renders publication assets.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from diffraction import (
    Grid,
    create_grid,
    circular_aperture,
    rectangular_aperture,
    double_slit_aperture,
    annular_aperture,
    gaussian_field,
    binary_grating,
    triangular_aperture,
    shifted_circular_aperture,
    propagate_asm,
    propagate_blas,
    propagate_fresnel,
    propagate_sas,
    propagate_rs_direct,
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    relative_l2_error,
    compute_shared_region_l2_error,
    compare_on_common_grid,
    sas_output_scaling,
    total_power,
    radial_profile,
    find_first_radial_minimum,
    thin_lens_phase,
    spiral_phase,
    topological_charge,
    gaussian_beam_abcd,
    check_sas_validity,
    measure_beam_radius,
    beam_waist_second_moment,
    asm_critical_distance,
    pad_field,
    asm_transfer_function,
    blas_transfer_function,
    compute_bandlimit_frequencies,
    jinc,
)
from visualize import (
    generate_fig01_aperture_diffraction,
    generate_fig02_methods_errors,
    generate_fig03_results_profiles,
    generate_fig04_method_selection_map,
    generate_fig05_thin_lens_focus,
    generate_fig06_vortex_beam,
    generate_primary_loop_gif,
    generate_error_loop_gif,
)


def run_validations(quick: bool = False) -> Dict[str, Any]:
    """Execute physical validation checks against analytical and numerical references."""
    results = {}
    wavelength = 532e-9

    # Test 1: Zero-distance identity
    grid_0 = create_grid(nx=128, ny=128, dx=5e-6, dy=5e-6)
    u_0 = circular_aperture(grid_0, radius=1.5e-4)
    u_asm_0 = propagate_asm(u_0, wavelength, z=0.0, dx=grid_0.dx, dy=grid_0.dy)
    u_blas_0 = propagate_blas(u_0, wavelength, z=0.0, dx=grid_0.dx, dy=grid_0.dy)
    u_fres_0 = propagate_fresnel(u_0, wavelength, z=0.0, dx=grid_0.dx, dy=grid_0.dy)
    u_sas_0, _ = propagate_sas(u_0, wavelength, z=0.0, dx=grid_0.dx, dy=grid_0.dy)

    err_asm_0 = relative_l2_error(u_asm_0, u_0)
    err_blas_0 = relative_l2_error(u_blas_0, u_0)
    err_fres_0 = relative_l2_error(u_fres_0, u_0)
    err_sas_0 = relative_l2_error(u_sas_0, u_0)

    results["zero_distance_identity"] = {
        "passed": bool(max(err_asm_0, err_blas_0, err_fres_0) < 1e-14),
        "error_asm": float(err_asm_0),
        "error_blas": float(err_blas_0),
        "error_fresnel": float(err_fres_0),
        "error_sas": float(err_sas_0),
        "note": "SAS at z=0 is a branch identity return (requires z > 0 for non-singular scaling)",
    }

    # Test 2: Analytical Airy disc validation at z = 0.50 m
    n_airy = 256 if quick else 512
    dx_airy = 10e-6 if quick else 5e-6
    radius = 1.5e-4
    z_airy = 0.50

    grid_airy = create_grid(nx=n_airy, ny=n_airy, dx=dx_airy, dy=dx_airy)
    u_circ_in = circular_aperture(grid_airy, radius=radius)
    u_blas_airy = propagate_blas(u_circ_in, wavelength, z=z_airy, dx=dx_airy, dy=dx_airy, padding=2)

    i_num = np.abs(u_blas_airy)**2
    i_ana = airy_pattern(grid_airy, wavelength, z=z_airy, radius=radius)

    r_num, prof_num = radial_profile(i_num, grid_airy.x, grid_airy.y, num_bins=150)
    r_ana, prof_ana = radial_profile(i_ana, grid_airy.x, grid_airy.y, num_bins=150)

    r1_theory = airy_first_zero(wavelength, z_airy, radius)
    r1_meas = find_first_radial_minimum(r_num, prof_num, search_range=(0.7 * r1_theory, 1.4 * r1_theory))
    r1_err_rel = abs(r1_meas - r1_theory) / r1_theory

    p_num_n = prof_num / np.max(prof_num)
    p_ana_n = prof_ana / np.max(prof_ana)
    prof_err = float(np.linalg.norm(p_num_n - p_ana_n) / np.linalg.norm(p_ana_n))

    results["circular_fraunhofer_airy"] = {
        "passed": bool(r1_err_rel < 0.02 and prof_err < 0.05),
        "r1_theory_m": float(r1_theory),
        "r1_measured_m": float(r1_meas),
        "r1_relative_error": float(r1_err_rel),
        "profile_l2_error": float(prof_err),
    }

    # Test 2b: Scalable Angular Spectrum Airy validation at z = 0.50 m
    u_sas_airy, grid_sas_airy = propagate_sas(u_circ_in, wavelength, z=z_airy, dx=dx_airy, dy=dx_airy, padding=2)
    i_sas_num = np.abs(u_sas_airy)**2
    i_sas_ana = airy_pattern(grid_sas_airy, wavelength, z=z_airy, radius=radius)

    r_sas_num, prof_sas_num = radial_profile(i_sas_num, grid_sas_airy.x, grid_sas_airy.y, num_bins=300)
    r_sas_ana, prof_sas_ana = radial_profile(i_sas_ana, grid_sas_airy.x, grid_sas_airy.y, num_bins=300)

    r1_sas_meas = find_first_radial_minimum(r_sas_num, prof_sas_num, search_range=(0.7 * r1_theory, 1.4 * r1_theory))
    r1_sas_err_rel = abs(r1_sas_meas - r1_theory) / r1_theory

    p_sas_num_n = prof_sas_num / np.max(prof_sas_num)
    p_sas_ana_n = prof_sas_ana / np.max(prof_sas_ana)
    prof_sas_err = float(np.linalg.norm(p_sas_num_n - p_sas_ana_n) / np.linalg.norm(p_sas_ana_n))

    results["scalable_angular_spectrum_airy"] = {
        "passed": bool(r1_sas_err_rel < 0.02 and prof_sas_err < 0.02),
        "r1_theory_m": float(r1_theory),
        "r1_measured_m": float(r1_sas_meas),
        "r1_relative_error": float(r1_sas_err_rel),
        "profile_l2_error": float(prof_sas_err),
    }

    # Test 3: Rectangular aperture sinc validation at z = 0.50 m
    w_rect, h_rect = 3.0e-4, 4.5e-4
    u_rect_in = rectangular_aperture(grid_airy, width=w_rect, height=h_rect)
    u_blas_rect = propagate_blas(u_rect_in, wavelength, z=z_airy, dx=dx_airy, dy=dx_airy, padding=2)

    i_rect_num = np.abs(u_blas_rect)**2
    i_rect_ana = sinc_pattern(grid_airy, wavelength, z=z_airy, width=w_rect, height=h_rect)

    cx, cy = n_airy // 2, n_airy // 2
    x_num = i_rect_num[cy, :] / np.max(i_rect_num[cy, :])
    x_ana = i_rect_ana[cy, :] / np.max(i_rect_ana[cy, :])
    y_num = i_rect_num[:, cx] / np.max(i_rect_num[:, cx])
    y_ana = i_rect_ana[:, cx] / np.max(i_rect_ana[:, cx])

    err_x = float(np.linalg.norm(x_num - x_ana) / np.linalg.norm(x_ana))
    err_y = float(np.linalg.norm(y_num - y_ana) / np.linalg.norm(y_ana))

    results["rectangular_fraunhofer_sinc"] = {
        "passed": bool(err_x < 0.05 and err_y < 0.05),
        "slice_x_error": float(err_x),
        "slice_y_error": float(err_y),
    }

    # Test 4: Near-field direct Rayleigh-Sommerfeld integration at z = 0.005 m
    grid_rs = create_grid(nx=64, ny=64, dx=8e-6, dy=8e-6)
    u_circ_rs = circular_aperture(grid_rs, radius=8e-5)
    z_nf = 0.005
    u_blas_nf = propagate_blas(u_circ_rs, wavelength, z=z_nf, dx=8e-6, dy=8e-6, padding=2)

    sel_x = np.linspace(16, 48, 16, dtype=int)
    sel_y = np.linspace(16, 48, 16, dtype=int)
    sub_x = grid_rs.x[sel_x]
    sub_y = grid_rs.y[sel_y]

    u_rs = propagate_rs_direct(u_circ_rs, wavelength, z=z_nf, dx=8e-6, dy=8e-6, x_out=sub_x, y_out=sub_y)
    u_blas_sub = u_blas_nf[np.ix_(sel_y, sel_x)]

    err_rs = float(np.linalg.norm(np.abs(u_blas_sub)**2 - np.abs(u_rs)**2) / np.linalg.norm(np.abs(u_rs)**2))

    results["near_field_direct_integration"] = {
        "passed": bool(err_rs < 0.05),
        "relative_l2_error": float(err_rs),
    }

    # Test 5: Thin-lens focusing against ABCD Gaussian beam theory
    n_lens = 256 if quick else 512
    dx_lens = 5e-6
    w0_lens = 3.0e-4
    f_lens = 75e-3

    grid_lens = create_grid(nx=n_lens, ny=n_lens, dx=dx_lens, dy=dx_lens)
    u_gauss_in = gaussian_field(grid_lens, waist=w0_lens)
    t_lens = thin_lens_phase(grid_lens, wavelength, f_lens)
    u_lens = u_gauss_in * t_lens

    _, _, z_focus_th, w_focus_th = gaussian_beam_abcd(w0_lens, wavelength, f_lens, z=f_lens)

    # Search axial range with dense z sweep
    z_scan_min = 70.0e-3
    z_scan_max = 77.0e-3
    nz_focus_scan = 29 if quick else 71
    z_scan = np.linspace(z_scan_min, z_scan_max, nz_focus_scan)
    waists_meas = []
    waists_th = []

    for z_i in z_scan:
        u_zi = propagate_blas(u_lens, wavelength, z=z_i, dx=dx_lens, dy=dx_lens, padding=2)
        # Calculate second-moment beam waist with physical pixel-area weighting
        w_m = measure_beam_radius(np.abs(u_zi)**2, grid_lens.x, grid_lens.y, roi_radius=4.0 * w_focus_th)
        w_t, _, _, _ = gaussian_beam_abcd(w0_lens, wavelength, f_lens, z_i)
        waists_meas.append(w_m)
        waists_th.append(w_t)

    min_idx = int(np.argmin(waists_meas))
    # Local quadratic fit (5 adjacent points) around the discrete minimum to find continuous focal minimum
    fit_slice = slice(max(0, min_idx - 2), min(len(z_scan), min_idx + 3))
    p_fit = np.polyfit(z_scan[fit_slice], np.array(waists_meas)[fit_slice], 2)
    z_focus_meas = float(-p_fit[1] / (2.0 * p_fit[0]))
    w_focus_meas = float(np.polyval(p_fit, z_focus_meas))

    # Evaluate true relative errors from raw unrounded measurements
    pos_err_rel = float(abs(z_focus_meas - z_focus_th) / z_focus_th)
    waist_err_rel = float(abs(w_focus_meas - w_focus_th) / w_focus_th)
    beam_radius_l2 = float(np.linalg.norm(np.array(waists_meas) - np.array(waists_th)) / np.linalg.norm(np.array(waists_th)))

    dz_step = float(z_scan[1] - z_scan[0])
    focus_uncertainty = dz_step / 10.0
    waist_uncertainty = 1.0e-7

    results["thin_lens_focus"] = {
        "passed": bool(pos_err_rel < 0.02 and waist_err_rel < 0.03),
        "focus_z_theory_m": float(z_focus_th),
        "focus_z_measured_m": float(z_focus_meas),
        "focus_z_relative_error": float(pos_err_rel),
        "focus_z_uncertainty_m": float(focus_uncertainty),
        "waist_theory_m": float(w_focus_th),
        "waist_measured_m": float(w_focus_meas),
        "waist_relative_error": float(waist_err_rel),
        "waist_uncertainty_m": float(waist_uncertainty),
        "beam_radius_l2_error": float(beam_radius_l2),
        "z_step_m": float(dz_step),
        "fit_method": "quadratic_interpolation",
        "precision_note": f"Focus position {z_focus_meas*1e3:.4f} mm (raw rel err {pos_err_rel*100:.4f}%), waist {w_focus_meas*1e6:.3f} um (raw rel err {waist_err_rel*100:.4f}%)",
    }

    # Test 6: Optical vortex topological charge conservation
    charges_to_test = [1, 2] if quick else [1, 2, 3, -1]
    radii_to_test = [1.5e-4] if quick else [1.2e-4, 1.5e-4, 1.8e-4]
    distances_to_test = [0.20] if quick else [0.10, 0.20, 0.30]
    vortex_results = []
    vortex_passed = True

    for ell in charges_to_test:
        t_sp = spiral_phase(grid_lens, charge=ell)
        u_v = u_gauss_in * t_sp

        # Measure charge on input across multiple radii
        q_in_runs = [topological_charge(u_v, grid_lens, radius=r, num_points=360) for r in radii_to_test]
        q_in = float(np.mean(q_in_runs))

        # Propagate to multiple distances and test across multiple radii
        q_out_all = []
        for z_v in distances_to_test:
            u_v_prop = propagate_blas(u_v, wavelength, z=z_v, dx=dx_lens, dy=dx_lens, padding=2)
            q_out_all.extend([topological_charge(u_v_prop, grid_lens, radius=r, num_points=360) for r in radii_to_test])

        q_out = float(np.mean(q_out_all))
        err_in = abs(q_in - ell)
        err_out = abs(q_out - ell)
        if err_in > 0.02 or err_out > 0.05:
            vortex_passed = False

        vortex_results.append({
            "charge_requested": int(ell),
            "charge_input_measured": float(q_in),
            "charge_output_measured": float(q_out),
            "error_input": float(err_in),
            "error_output": float(err_out),
            "radii_tested_m": [float(r) for r in radii_to_test],
            "distances_tested_m": [float(d) for d in distances_to_test],
        })

    results["optical_vortex"] = {
        "passed": bool(vortex_passed),
        "charges": vortex_results,
    }

    return results


def run_method_selection_sweep(quick: bool = False) -> List[Dict[str, Any]]:
    """Execute propagation method selection sweep across sampling pitch and distance."""
    wavelength = 532e-9
    aperture_radius = 1.5e-4  # D = 0.30 mm

    n_primary = 256 if quick else 512
    dx_values = [2.5e-6, 5.0e-6, 10.0e-6] if quick else [2.5e-6, 5.0e-6, 7.5e-6, 10.0e-6, 15.0e-6]
    z_values = np.geomspace(5e-3, 1.0, 6) if quick else np.geomspace(5e-3, 1.0, 14)

    metrics_rows = []
    methods = ["fresnel", "asm", "blas", "sas"]

    for dx in dx_values:
        grid_in = create_grid(nx=n_primary, ny=n_primary, dx=dx, dy=dx)
        u_in = circular_aperture(grid_in, radius=aperture_radius)
        source_fov = grid_in.Sx
        area_discrete = float(np.sum(np.abs(u_in)**2) * dx * dx)

        # 4x padded reference grid and padded field
        grid_ref_4x = create_grid(nx=n_primary * 4, ny=n_primary * 4, dx=dx, dy=dx)
        padded_4x, _ = pad_field(u_in, factor=4)
        spec_ref_4x = fftshift(fft2(ifftshift(padded_4x)))

        # 2x padded reference grid for convergence check
        grid_ref_2x = create_grid(nx=n_primary * 2, ny=n_primary * 2, dx=dx, dy=dx)
        padded_2x, _ = pad_field(u_in, factor=2)
        spec_ref_2x = fftshift(fft2(ifftshift(padded_2x)))

        zc_4x = asm_critical_distance(n_primary, dx, wavelength, padding=4)

        for z in z_values:
            # Physical diffraction ROI at distance z
            r1_airy = 1.22 * wavelength * z / (2.0 * aperture_radius)
            w_target = max(source_fov, 2.0 * r1_airy)

            # Determine appropriate physical ground truth reference per regime:
            # A. Far field (z >= 0.35 m, checked Fraunhofer validity N_F = a^2 / (lambda*z) <= 0.12):
            # Analytical Fraunhofer Airy pattern with amplitude normalized to discrete pupil transmission area.
            # Avoids transfer-function aliasing completely and serves as an independent asymptotic reference
            # with ~0.5% parabolic phase truncation uncertainty (not universally exact for all z).
            if z >= 0.35:
                k = 2.0 * np.pi / wavelength
                r_ref = np.sqrt(grid_ref_4x.X**2 + grid_ref_4x.Y**2)
                q_ref = k * aperture_radius * r_ref / z
                amp_ref = (area_discrete / (wavelength * z)) * jinc(q_ref)
                i_ref = amp_ref**2
                x_ref_arr = grid_ref_4x.x
                y_ref_arr = grid_ref_4x.y
                reference_converged = True
                ref_type = "analytical_airy"
                ref_uncertainty = 0.005  # Asymptotic Fraunhofer parabolic truncation uncertainty ~0.5%

            # B. Near field: requires z <= zc_4x (strictly evaluated for this dx) AND z <= 0.05 m:
            elif z <= 0.05 and z <= zc_4x:
                h_ref_4x = asm_transfer_function(grid_ref_4x, wavelength, z)
                u_ref_4x = fftshift(ifft2(ifftshift(spec_ref_4x * h_ref_4x)))
                i_ref = np.abs(u_ref_4x)**2
                x_ref_arr = grid_ref_4x.x
                y_ref_arr = grid_ref_4x.y

                h_ref_2x = asm_transfer_function(grid_ref_2x, wavelength, z)
                u_ref_2x = fftshift(ifft2(ifftshift(spec_ref_2x * h_ref_2x)))
                diff_ref, _, _ = compare_on_common_grid(
                    u_ref_2x, grid_ref_2x.x, grid_ref_2x.y,
                    i_ref, grid_ref_4x.x, grid_ref_4x.y,
                    n_eval=80
                )
                reference_converged = bool(diff_ref <= 0.03)
                ref_type = "near_field_asm_4x" if reference_converged else "reference_unresolved"
                ref_uncertainty = float(diff_ref)

            # C. Intermediate regime:
            else:
                # Check BL-ASM cutoff frequency relative to significant pupil spectral extent:
                # For circular pupil of diameter D = 2a, the angular spectrum zeros occur at u_n / (pi * D).
                # The 3rd dark ring is at u_3 / (pi * D) ~ 10.17 / (pi * D) ~ 3.24 / D.
                # Threshold f_pupil = 3.0 / D = 3.0 / (2 * a) = 10,000 m^-1 encloses > 95% of pupil spectral energy.
                # If fx_lim < f_pupil, BL-ASM bandlimits the physical pupil diffraction lobes themselves.
                fx_lim_4x, _ = compute_bandlimit_frequencies(wavelength, z, grid_ref_4x.Sx, grid_ref_4x.Sy)
                f_pupil_threshold = 3.0 / (2.0 * aperture_radius)  # 3/D = 10,000 m^-1 for D=0.30 mm

                h_blas_4x, _ = blas_transfer_function(grid_ref_4x, wavelength, z)
                u_blas_4x = fftshift(ifft2(ifftshift(spec_ref_4x * h_blas_4x)))
                i_ref = np.abs(u_blas_4x)**2
                x_ref_arr = grid_ref_4x.x
                y_ref_arr = grid_ref_4x.y

                h_blas_2x, _ = blas_transfer_function(grid_ref_2x, wavelength, z)
                u_blas_2x = fftshift(ifft2(ifftshift(spec_ref_2x * h_blas_2x)))
                diff_ref, _, _ = compare_on_common_grid(
                    u_blas_2x, grid_ref_2x.x, grid_ref_2x.y,
                    i_ref, grid_ref_4x.x, grid_ref_4x.y,
                    n_eval=80
                )
                # If padding changes field by > 3% or cutoff frequency truncates physical pupil spectrum, keep unresolved
                reference_converged = bool(diff_ref <= 0.03 and fx_lim_4x >= f_pupil_threshold)
                ref_type = "intermediate_blas_4x" if reference_converged else "reference_unresolved"
                ref_uncertainty = float(diff_ref)

            for method in methods:
                # Timed benchmarking with warmup
                if method == "fresnel":
                    _ = propagate_fresnel(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        u_out = propagate_fresnel(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                        times.append(time.perf_counter() - t0)
                    grid_out = grid_in
                    zc_2x = asm_critical_distance(n_primary, dx, wavelength, padding=2)
                    is_valid = bool(z <= zc_2x * 2.0)
                    sas_reason = "Valid Fresnel" if is_valid else "Transfer-function aliased"
                    mag = 1.0

                elif method == "asm":
                    _ = propagate_asm(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        u_out = propagate_asm(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                        times.append(time.perf_counter() - t0)
                    grid_out = grid_in
                    zc_2x = asm_critical_distance(n_primary, dx, wavelength, padding=2)
                    is_valid = bool(z <= zc_2x)
                    sas_reason = "Valid fixed-grid ASM" if is_valid else f"Chirp aliased (z > z_c={zc_2x*1e3:.0f} mm)"
                    mag = 1.0

                elif method == "blas":
                    _ = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        u_out = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                        times.append(time.perf_counter() - t0)
                    grid_out = grid_in
                    is_valid = True
                    sas_reason = "Valid band-limited spectrum"
                    mag = 1.0

                else:  # sas
                    _ = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        u_out, grid_out = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dx, padding=2)
                        times.append(time.perf_counter() - t0)
                    is_valid, sas_reason, mag = check_sas_validity(wavelength, z, n_primary, n_primary, dx, dx, padding=2)

                med_runtime = float(np.median(times))

                # Physical field of view coverage relative to target diffraction ROI
                cov_x = float(min(1.0, grid_out.Sx / w_target))
                cov_y = float(min(1.0, grid_out.Sy / w_target))
                coverage_2d = float(cov_x * cov_y)

                # Reference optical power captured inside method output FOV
                mask_ref_in_method = (np.abs(grid_ref_4x.X) <= grid_out.Sx / 2.0) & (np.abs(grid_ref_4x.Y) <= grid_out.Sy / 2.0)
                captured_power_ratio = float(np.sum(i_ref[mask_ref_in_method]) / max(1e-15, np.sum(i_ref)))

                # Strict coverage criterion: requires covering >= 95% of target linear extent and >= 90% 2D extent
                sufficient_coverage = bool(cov_x >= 0.95 and coverage_2d >= 0.90)

                # Physical coordinate comparison on common overlap Omega_common
                rel_l2, peak_l2, common_w = compare_on_common_grid(
                    u_out, grid_out.x, grid_out.y,
                    i_ref, x_ref_arr, y_ref_arr,
                    n_eval=160
                )

                metrics_rows.append({
                    "experiment": "method_selection_map",
                    "N": n_primary,
                    "method": method,
                    "z_m": round(float(z), 6),
                    "dx_source_m": round(float(dx), 8),
                    "dx_output_m": round(float(grid_out.dx), 8),
                    "source_fov_m": round(float(source_fov), 6),
                    "output_fov_m": round(float(grid_out.Sx), 6),
                    "magnification": round(float(mag), 4),
                    "common_width_m": round(float(common_w), 6),
                    "coverage_ratio": round(float(cov_x), 4),
                    "coverage_2d": round(float(coverage_2d), 4),
                    "captured_power_ratio": round(float(captured_power_ratio), 4),
                    "sufficient_coverage": bool(sufficient_coverage),
                    "relative_l2_error": round(float(rel_l2), 6),
                    "peak_normalized_error": round(float(peak_l2), 6),
                    "runtime_s": round(float(med_runtime), 6),
                    "valid": bool(is_valid),
                    "sas_reason": sas_reason,
                    "reference_converged": bool(reference_converged),
                    "ref_type": ref_type,
                    "ref_uncertainty": round(float(ref_uncertainty), 6),
                })

    return metrics_rows


def run_benchmark_metrics(quick: bool = False) -> List[Dict[str, Any]]:
    """Generate fair metrics comparing propagation methods on shared physical coordinates."""
    wavelength = 532e-9
    radius = 1.5e-4
    z_eval = 0.50
    metrics_rows = []

    # 1. Primary method selection sweep
    map_rows = run_method_selection_sweep(quick=quick)
    metrics_rows.extend(map_rows)

    # 2. Experiment: Sampling sweep with fixed input window (S_in = 2.56 mm)
    resolutions = [256, 512] if quick else [256, 512, 1024]
    fov_in = 2.56e-3

    for n in resolutions:
        dx = fov_in / n
        grid = create_grid(nx=n, ny=n, dx=dx, dy=dx)
        u_in = circular_aperture(grid, radius=radius)
        i_ref = airy_pattern(grid, wavelength, z=z_eval, radius=radius)

        for method in ["fresnel", "asm", "blas", "sas"]:
            t0 = time.perf_counter()
            if method == "fresnel":
                u_out = propagate_fresnel(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
                grid_out = grid
            elif method == "asm":
                u_out = propagate_asm(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
                grid_out = grid
            elif method == "blas":
                u_out = propagate_blas(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
                grid_out = grid
            else:
                u_out, grid_out = propagate_sas(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
            t_elapsed = time.perf_counter() - t0

            i_out = np.abs(u_out)**2

            if method == "sas":
                i_ref_sas = airy_pattern(grid_out, wavelength, z=z_eval, radius=radius)
                fov_out = grid_out.Sx
                dx_out = grid_out.dx
                coverage = f"Scaled ({fov_out*1e3:.1f} mm)"
                shared_raw, shared_peak, common_w = compare_on_common_grid(
                    i_out, grid_out.x, grid_out.y, i_ref_sas, grid_out.x, grid_out.y, n_eval=160
                )
                mag = grid_out.dx / dx
            else:
                fov_out = fov_in
                dx_out = dx
                coverage = f"Shared ({fov_out*1e3:.2f} mm)"
                shared_raw = relative_l2_error(i_out, i_ref)
                shared_peak = relative_l2_error(i_out / np.max(i_out), i_ref / np.max(i_ref))
                common_w = fov_in
                mag = 1.0

            metrics_rows.append({
                "experiment": "fixed_fov",
                "N": n,
                "method": method,
                "z_m": z_eval,
                "dx_source_m": round(dx, 8),
                "dx_output_m": round(dx_out, 8),
                "source_fov_m": round(fov_in, 6),
                "output_fov_m": round(fov_out, 6),
                "magnification": round(mag, 4),
                "common_width_m": round(common_w, 6),
                "relative_l2_error": round(shared_raw, 6),
                "peak_normalized_error": round(shared_peak, 6),
                "runtime_s": round(t_elapsed, 4),
                "valid": True,
                "sas_reason": coverage,
                "reference_converged": True,
            })

    return metrics_rows


def save_results(validations: Dict[str, Any], metrics: List[Dict[str, Any]]) -> None:
    """Save validation dictionary to JSON and benchmark metrics to CSV."""
    os.makedirs("results", exist_ok=True)

    with open("results/validation.json", "w", encoding="utf-8") as f:
        json.dump(validations, f, indent=2)

    with open("results/metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    parser = argparse.ArgumentParser(description="Reproduce scalar diffraction experiments.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification with smaller grids.")
    parser.add_argument("--figure-only", action="store_true", help="Only generate publication figures 1 to 6.")
    parser.add_argument("--gif-only", action="store_true", help="Only generate animated loop GIFs.")
    parser.add_argument("--all", action="store_true", help="Run validation, benchmarks, and all figures and GIFs.")
    args = parser.parse_args()

    if args.figure_only:
        print("Generating Figures 1 through 6...")
        generate_fig01_aperture_diffraction()
        generate_fig02_methods_errors()
        generate_fig03_results_profiles()
        generate_fig04_method_selection_map()
        generate_fig05_thin_lens_focus()
        generate_fig06_vortex_beam()
        print("Figures saved in assets/")
        return

    if args.gif_only:
        print("Generating primary diffraction loop GIF...")
        n1 = generate_primary_loop_gif()
        print(f"Primary GIF saved to assets/diffraction_loop.gif ({n1} frames).")
        print("Generating error evolution loop GIF...")
        n2 = generate_error_loop_gif()
        print(f"Error GIF saved to assets/error_loop.gif ({n2} frames).")
        return

    print("Running physical validation tests...")
    validations = run_validations(quick=args.quick)
    all_passed = all(v["passed"] for v in validations.values())
    status_str = "PASS" if all_passed else "FAIL"
    print(f"Validation status: {status_str}")

    if not all_passed:
        failing_tests = [k for k, v in validations.items() if not v["passed"]]
        print(f"ERROR: Mandatory physical validation FAILED in checks: {failing_tests}", file=sys.stderr)
        save_results(validations, [])
        sys.exit(1)

    print("Benchmarking propagation methods and sampling convergence...")
    metrics = run_benchmark_metrics(quick=args.quick)
    save_results(validations, metrics)
    print("Metrics written to results/metrics.csv and results/validation.json")

    print("Rendering Figures 1 through 6...")
    generate_fig01_aperture_diffraction()
    generate_fig02_methods_errors()
    generate_fig03_results_profiles()
    generate_fig04_method_selection_map(metrics_data=metrics, quick=args.quick)
    generate_fig05_thin_lens_focus(lens_data=validations.get("thin_lens_focus"), quick=args.quick)
    generate_fig06_vortex_beam(vortex_data=validations.get("optical_vortex"), quick=args.quick)
    print("Figures 1 through 6 rendered successfully.")

    if not args.quick or args.all:
        print("Rendering animated GIFs...")
        n1 = generate_primary_loop_gif()
        print(f"Primary GIF saved ({n1} frames).")
        n2 = generate_error_loop_gif()
        print(f"Error GIF saved ({n2} frames).")
    else:
        print("Skipping animated GIF rendering in --quick mode (GIFs preserved in assets/).")

    print("All requested reproduction tasks completed successfully.")


if __name__ == "__main__":
    main()
