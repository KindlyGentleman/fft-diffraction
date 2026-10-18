"""Reproduction CLI runner for FFT scalar diffraction experiments.

Runs validation tests, generates metrics tables, and renders publication assets.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import numpy as np
import scipy
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
    characterize_thin_lens_focus,
    check_sas_validity,
    measure_beam_radius,
    beam_waist_second_moment,
    asm_critical_distance,
    pad_field,
    asm_transfer_function,
    blas_transfer_function,
    compute_bandlimit_frequencies,
    compute_complex_field_error,
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


def _compute_sha256(filepath: str) -> str:
    """Compute hex SHA256 digest of a file for integrity manifests."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_source_provenance() -> Dict[str, Any]:
    """Compute deterministic source provenance including commit, dirty status, and source digest."""
    git_rev = "unknown"
    is_dirty = False
    modified_files = []

    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        git_rev = res.stdout.strip()
    except Exception:
        pass

    try:
        diff_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        status_lines = [line.strip() for line in diff_res.stdout.splitlines() if line.strip()]
        source_mods = [line for line in status_lines if any(src in line for src in ("diffraction.py", "reproduce.py", "visualize.py"))]
        is_dirty = len(source_mods) > 0
        modified_files = [line.split()[-1] for line in status_lines]
    except Exception:
        pass

    source_hasher = hashlib.sha256()
    core_files = ["diffraction.py", "reproduce.py", "visualize.py"]
    source_file_hashes = {}
    for sf in core_files:
        if os.path.exists(sf):
            h = _compute_sha256(sf)
            source_file_hashes[sf] = h
            source_hasher.update(h.encode("utf-8"))
    source_digest = source_hasher.hexdigest()

    return {
        "git_commit": git_rev,
        "is_dirty": is_dirty,
        "modified_files": modified_files,
        "source_digest": source_digest,
        "source_file_hashes": source_file_hashes,
    }


def verify_results_integrity(output_dir: str = "results") -> bool:
    """Verify that output_dir contains complete, untampered validation, metrics, and manifest files."""
    man_path = os.path.join(output_dir, "manifest.json")
    if not os.path.isfile(man_path):
        return False
    try:
        with open(man_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for fname in ("validation.json", "metrics.csv"):
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                return False
            expected_sha = manifest["files"][fname]["sha256"]
            actual_sha = _compute_sha256(fpath)
            if actual_sha != expected_sha:
                return False
        return True
    except Exception:
        return False


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
    err_complex_rs, opt_phase_rs = compute_complex_field_error(u_blas_sub, u_rs)

    # Acceptance gating: intensity relative L2 < 5% and phase-aligned complex-field error < 10%
    # Tolerance is physically justified by numerical quadrature discretization of direct RS integral on 64x64 grid
    passed_intensity = bool(err_rs < 0.05)
    passed_complex = bool(err_complex_rs < 0.10)
    results["near_field_direct_integration"] = {
        "passed": bool(passed_intensity and passed_complex),
        "passed_intensity": passed_intensity,
        "passed_complex": passed_complex,
        "relative_l2_error": float(err_rs),
        "complex_field_error": float(err_complex_rs),
        "optimal_phase_alignment_rad": float(opt_phase_rs),
        "quadrature_grid": "64x64",
    }

    # Test 5: Thin-lens focusing against ABCD Gaussian beam theory
    # Evaluated via unified characterize_thin_lens_focus pipeline without pre-rounding
    res_lens = characterize_thin_lens_focus(
        w0=3.0e-4,
        wavelength=wavelength,
        focal_length=75e-3,
        nx=256 if quick else 512,
        ny=256 if quick else 512,
        dx=5e-6,
        dy=5e-6,
        z_scan_min=70.0e-3,
        z_scan_max=77.0e-3,
        n_z_samples=29 if quick else 71,
        fit_window_points=5,
    )
    results["thin_lens_focus"] = res_lens

    # Test 6: Optical vortex topological charge conservation
    n_vortex = 256 if quick else 512
    dx_vortex = 5e-6
    w0_vortex = 3.0e-4
    grid_vortex = create_grid(nx=n_vortex, ny=n_vortex, dx=dx_vortex, dy=dx_vortex)
    u_gauss_vortex = gaussian_field(grid_vortex, waist=w0_vortex)

    charges_to_test = [1, 2] if quick else [1, 2, 3, -1]
    radii_to_test = [1.5e-4] if quick else [1.2e-4, 1.5e-4, 1.8e-4]
    distances_to_test = [0.20] if quick else [0.10, 0.20, 0.30]
    vortex_results = []
    vortex_passed = True

    for ell in charges_to_test:
        t_sp = spiral_phase(grid_vortex, charge=ell)
        u_v = u_gauss_vortex * t_sp

        # Measure charge on input across multiple radii with individual contour gating
        q_in_runs = []
        amp_in_runs = []
        for r in radii_to_test:
            q_meas, min_amp = topological_charge(u_v, grid_vortex, radius=r, num_points=360, return_min_amplitude=True)
            q_in_runs.append(q_meas)
            amp_in_runs.append(min_amp)
            # Gate every individual contour independently: phase accuracy and amplitude away from numerical null
            if abs(q_meas - ell) > 0.02 or min_amp < 1e-12:
                vortex_passed = False

        q_in = float(np.mean(q_in_runs))
        max_err_in = float(max(abs(q - ell) for q in q_in_runs))

        # Propagate to multiple distances and test across multiple radii
        q_out_all = []
        amp_out_all = []
        max_err_out = 0.0
        for z_v in distances_to_test:
            u_v_prop = propagate_blas(u_v, wavelength, z=z_v, dx=dx_vortex, dy=dx_vortex, padding=2)
            for r in radii_to_test:
                q_meas, min_amp = topological_charge(u_v_prop, grid_vortex, radius=r, num_points=360, return_min_amplitude=True)
                q_out_all.append(q_meas)
                amp_out_all.append(min_amp)
                err_c = abs(q_meas - ell)
                if err_c > max_err_out:
                    max_err_out = err_c
                if err_c > 0.05 or min_amp < 1e-12:
                    vortex_passed = False

        q_out = float(np.mean(q_out_all))
        vortex_results.append({
            "charge_requested": int(ell),
            "charge_input_measured": float(q_in),
            "charge_output_measured": float(q_out),
            "error_input_mean": float(abs(q_in - ell)),
            "error_output_mean": float(abs(q_out - ell)),
            "max_contour_error_input": float(max_err_in),
            "max_contour_error_output": float(max_err_out),
            "min_amplitude_observed": float(min(min(amp_in_runs), min(amp_out_all))),
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

                # Optical power metrics:
                # 1. Total beam power captured inside method output FOV relative to total physical input beam power
                p_out = float(np.sum(np.abs(u_out)**2) * grid_out.dx * grid_out.dy)
                captured_power_ratio = float(p_out / max(1e-15, area_discrete))

                # 2. Reference field power enclosed inside method's output spatial window
                mask_ref_in_method = (np.abs(grid_ref_4x.X) <= grid_out.Sx / 2.0) & (np.abs(grid_ref_4x.Y) <= grid_out.Sy / 2.0)
                ref_win_power_ratio = float(np.sum(i_ref[mask_ref_in_method]) / max(1e-15, np.sum(i_ref)))

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
                    "ref_window_power_ratio": round(float(ref_win_power_ratio), 4),
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
        p_in_ff = float(np.sum(np.abs(u_in)**2) * dx * dx)

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
            p_out_ff = float(np.sum(i_out) * grid_out.dx * grid_out.dy)
            cap_p_ratio = float(p_out_ff / max(1e-15, p_in_ff))

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
                "coverage_ratio": 1.0,
                "coverage_2d": 1.0,
                "captured_power_ratio": round(cap_p_ratio, 4),
                "ref_window_power_ratio": 1.0,
                "sufficient_coverage": True,
                "relative_l2_error": round(shared_raw, 6),
                "peak_normalized_error": round(shared_peak, 6),
                "runtime_s": round(t_elapsed, 4),
                "valid": True,
                "sas_reason": coverage,
                "reference_converged": True,
                "ref_type": "analytical_airy",
                "ref_uncertainty": 0.005,
            })

    return metrics_rows


BENCHMARK_COLUMNS = [
    "experiment", "N", "method", "z_m", "dx_source_m", "dx_output_m",
    "source_fov_m", "output_fov_m", "magnification", "common_width_m",
    "coverage_ratio", "coverage_2d", "captured_power_ratio", "ref_window_power_ratio",
    "sufficient_coverage", "relative_l2_error", "peak_normalized_error", "runtime_s",
    "valid", "sas_reason", "reference_converged", "ref_type", "ref_uncertainty"
]


def save_results(
    validations: Dict[str, Any],
    metrics: List[Dict[str, Any]],
    output_dir: str = "results",
    quick: bool = False,
    overwrite_full: bool = False,
) -> Dict[str, Any]:
    """Save validation dictionary, benchmark CSV, and manifest atomically via transactional staging.

    Guarantees:
    1. Transactional staging: files are written and hashed in a staging directory.
    2. Run archiving: every execution is archived to output_dir/runs/<run_id>/.
    3. Quick-mode protection: if existing top-level results are from a full run,
       running --quick will preserve full results unless --overwrite-full is passed.
    4. Group publication: manifest.json contains SHA256 hashes of all published files.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    staging_dir = os.path.join(output_dir, f".staging_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging_dir, exist_ok=True)

    try:
        val_path_stage = os.path.join(staging_dir, "validation.json")
        with open(val_path_stage, "w", encoding="utf-8", newline="\n") as f:
            json.dump(validations, f, indent=2)

        met_path_stage = os.path.join(staging_dir, "metrics.csv")
        if metrics:
            fieldnames = list(metrics[0].keys())
            with open(met_path_stage, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(metrics)
        else:
            with open(met_path_stage, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLUMNS, lineterminator="\n")
                writer.writeheader()

        val_sha = _compute_sha256(val_path_stage)
        met_sha = _compute_sha256(met_path_stage)
        val_bytes = os.path.getsize(val_path_stage)
        met_bytes = os.path.getsize(met_path_stage)

        provenance = _compute_source_provenance()
        git_rev = provenance.get("git_commit", "unknown")

        all_val_passed = bool(validations) and all(v.get("passed", False) for v in validations.values())

        manifest = {
            "schema_version": "2.0.0",
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_mode": "quick" if quick else "full",
            "git_commit": git_rev,
            "is_dirty": provenance.get("is_dirty", False),
            "source_digest": provenance.get("source_digest", ""),
            "source_file_hashes": provenance.get("source_file_hashes", {}),
            "platform": {
                "system": platform.platform(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "files": {
                "validation.json": {
                    "filename": "validation.json",
                    "sha256": val_sha,
                    "size_bytes": val_bytes,
                    "num_checks": len(validations),
                    "all_passed": all_val_passed,
                },
                "metrics.csv": {
                    "filename": "metrics.csv",
                    "sha256": met_sha,
                    "size_bytes": met_bytes,
                    "num_records": len(metrics),
                },
            },
            "all_validations_passed": all_val_passed,
        }

        manifest_path_stage = os.path.join(staging_dir, "manifest.json")
        with open(manifest_path_stage, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)

        active_run_stage = os.path.join(staging_dir, "active_run.json")
        active_run_data = {
            "run_id": run_id,
            "timestamp_utc": manifest["timestamp_utc"],
            "execution_mode": manifest["execution_mode"],
            "git_commit": git_rev,
            "is_dirty": provenance.get("is_dirty", False),
            "source_digest": provenance.get("source_digest", ""),
            "all_validations_passed": all_val_passed,
        }
        with open(active_run_stage, "w", encoding="utf-8", newline="\n") as f:
            json.dump(active_run_data, f, indent=2)

        # 1. Archive run to output_dir/runs/<run_id>/
        run_archive_dir = os.path.join(output_dir, "runs", run_id)
        os.makedirs(run_archive_dir, exist_ok=True)
        shutil.copy2(val_path_stage, os.path.join(run_archive_dir, "validation.json"))
        shutil.copy2(met_path_stage, os.path.join(run_archive_dir, "metrics.csv"))
        shutil.copy2(manifest_path_stage, os.path.join(run_archive_dir, "manifest.json"))
        shutil.copy2(active_run_stage, os.path.join(run_archive_dir, "active_run.json"))

        # 2. Check quick-mode protection for top-level results
        top_manifest_path = os.path.join(output_dir, "manifest.json")
        existing_mode_full = False
        if os.path.exists(top_manifest_path):
            try:
                with open(top_manifest_path, "r", encoding="utf-8") as f:
                    top_m = json.load(f)
                    existing_mode_full = (top_m.get("execution_mode") == "full")
            except Exception:
                pass

        if quick and existing_mode_full and not overwrite_full:
            # Preserve existing top-level full benchmark; save quick outputs to *_quick.*
            quick_val = os.path.join(output_dir, "validation_quick.json")
            quick_met = os.path.join(output_dir, "metrics_quick.csv")
            quick_man = os.path.join(output_dir, "manifest_quick.json")
            quick_act = os.path.join(output_dir, "active_run_quick.json")
            shutil.copy2(val_path_stage, quick_val + ".tmp")
            shutil.copy2(met_path_stage, quick_met + ".tmp")
            shutil.copy2(manifest_path_stage, quick_man + ".tmp")
            shutil.copy2(active_run_stage, quick_act + ".tmp")
            os.replace(quick_val + ".tmp", quick_val)
            os.replace(quick_met + ".tmp", quick_met)
            os.replace(quick_man + ".tmp", quick_man)
            os.replace(quick_act + ".tmp", quick_act)
            print(
                f"[INFO] Preserved existing full benchmark results in {output_dir}/. "
                f"Quick results saved to runs/{run_id}/ and {output_dir}/*_quick.*. "
                f"Pass --overwrite-full to replace top-level files.",
                flush=True
            )
        else:
            # Atomically publish to top-level output_dir
            val_top = os.path.join(output_dir, "validation.json")
            met_top = os.path.join(output_dir, "metrics.csv")
            man_top = os.path.join(output_dir, "manifest.json")
            act_top = os.path.join(output_dir, "active_run.json")
            shutil.copy2(val_path_stage, val_top + ".tmp")
            shutil.copy2(met_path_stage, met_top + ".tmp")
            shutil.copy2(manifest_path_stage, man_top + ".tmp")
            shutil.copy2(active_run_stage, act_top + ".tmp")
            os.replace(val_top + ".tmp", val_top)
            os.replace(met_top + ".tmp", met_top)
            os.replace(man_top + ".tmp", man_top)
            os.replace(act_top + ".tmp", act_top)

        return manifest

    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce scalar diffraction experiments.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification with smaller grids.")
    parser.add_argument("--figure-only", action="store_true", help="Only generate publication figures 1 to 6.")
    parser.add_argument("--gif-only", action="store_true", help="Only generate animated loop GIFs.")
    parser.add_argument("--all", action="store_true", help="Run validation, benchmarks, and all figures and GIFs.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save numerical results.")
    parser.add_argument("--asset-dir", type=str, default="assets", help="Directory to save figures and animations.")
    parser.add_argument("--overwrite-full", action="store_true", help="Allow --quick mode to overwrite existing full reproduction results in top-level output directory.")
    args = parser.parse_args()

    if args.figure_only:
        print("Generating Figures 1 through 6...")
        generate_fig01_aperture_diffraction(asset_dir=args.asset_dir)
        generate_fig02_methods_errors(asset_dir=args.asset_dir)
        generate_fig03_results_profiles(asset_dir=args.asset_dir)
        generate_fig04_method_selection_map(asset_dir=args.asset_dir)
        generate_fig05_thin_lens_focus(asset_dir=args.asset_dir)
        generate_fig06_vortex_beam(asset_dir=args.asset_dir)
        print(f"Figures saved in {args.asset_dir}/")
        return

    if args.gif_only:
        print("Generating primary diffraction loop GIF...")
        n1 = generate_primary_loop_gif(asset_dir=args.asset_dir)
        print(f"Primary GIF saved to {args.asset_dir}/diffraction_loop.gif ({n1} frames).")
        print("Generating error evolution loop GIF...")
        n2 = generate_error_loop_gif(asset_dir=args.asset_dir)
        print(f"Error GIF saved to {args.asset_dir}/error_loop.gif ({n2} frames).")
        return

    print("Running physical validation tests...")
    validations = run_validations(quick=args.quick)
    all_passed = bool(validations) and all(v.get("passed", False) for v in validations.values())
    status_str = "PASS" if all_passed else "FAIL"
    print(f"Validation status: {status_str}")

    if not all_passed:
        failing_tests = [k for k, v in validations.items() if not v.get("passed", False)]
        print(f"ERROR: Mandatory physical validation FAILED in checks: {failing_tests}", file=sys.stderr)
        os.makedirs(args.output_dir, exist_ok=True)
        diag_path = os.path.join(args.output_dir, "validation_failure.json")
        try:
            with open(diag_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(validations, f, indent=2)
        except Exception:
            pass
        sys.exit(1)

    print("Benchmarking propagation methods and sampling convergence...")
    metrics = run_benchmark_metrics(quick=args.quick)
    manifest = save_results(validations, metrics, output_dir=args.output_dir, quick=args.quick, overwrite_full=args.overwrite_full)
    print(f"Metrics written to {args.output_dir}/ (run ID: {manifest['run_id']})")

    print("Rendering Figures 1 through 6...")
    generate_fig01_aperture_diffraction(asset_dir=args.asset_dir)
    generate_fig02_methods_errors(asset_dir=args.asset_dir)
    generate_fig03_results_profiles(asset_dir=args.asset_dir)
    generate_fig04_method_selection_map(metrics_data=metrics, quick=args.quick, asset_dir=args.asset_dir)
    generate_fig05_thin_lens_focus(lens_data=validations.get("thin_lens_focus"), quick=args.quick, asset_dir=args.asset_dir)
    generate_fig06_vortex_beam(vortex_data=validations.get("optical_vortex"), quick=args.quick, asset_dir=args.asset_dir)
    print("Figures 1 through 6 rendered successfully.")

    if not args.quick or args.all:
        print("Rendering animated GIFs...")
        n1 = generate_primary_loop_gif(asset_dir=args.asset_dir)
        print(f"Primary GIF saved ({n1} frames).")
        n2 = generate_error_loop_gif(asset_dir=args.asset_dir)
        print(f"Error GIF saved ({n2} frames).")
    else:
        print(f"Skipping animated GIF rendering in --quick mode (GIFs preserved in {args.asset_dir}/).")

    print("All requested reproduction tasks completed successfully.")


if __name__ == "__main__":
    main()
