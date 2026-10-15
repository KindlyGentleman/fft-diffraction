"""Reproduction CLI runner for FFT scalar diffraction experiments.

Runs validation tests, generates metrics tables, and renders publication assets.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

import numpy as np

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
)
from visualize import (
    generate_fig01_aperture_diffraction,
    generate_fig02_methods_errors,
    generate_fig03_results_profiles,
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

    err_asm_0 = relative_l2_error(u_asm_0, u_0)
    err_blas_0 = relative_l2_error(u_blas_0, u_0)
    err_fres_0 = relative_l2_error(u_fres_0, u_0)

    results["zero_distance_identity"] = {
        "passed": bool(max(err_asm_0, err_blas_0, err_fres_0) < 1e-14),
        "error_asm": float(err_asm_0),
        "error_blas": float(err_blas_0),
        "error_fresnel": float(err_fres_0),
    }

    # Test 2: Analytical Airy disc validation at z = 0.50 m
    n_airy = 256 if quick else 512
    dx_airy = 5e-6
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

    return results


def run_benchmark_metrics() -> List[Dict[str, Any]]:
    """Generate metrics comparing propagation methods across resolutions and distances."""
    wavelength = 532e-9
    radius = 1.5e-4
    z_eval = 0.50
    metrics_rows = []

    # Sampling sweep: Fixed FOV (S = 2.56 mm)
    resolutions = [256, 512, 1024]
    fov = 2.56e-3

    for n in resolutions:
        dx = fov / n
        grid = create_grid(nx=n, ny=n, dx=dx, dy=dx)
        u_in = circular_aperture(grid, radius=radius)
        i_ref = airy_pattern(grid, wavelength, z=z_eval, radius=radius)

        for method in ["fresnel", "asm", "blas"]:
            t0 = time.perf_counter()
            if method == "fresnel":
                u_out = propagate_fresnel(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
            elif method == "asm":
                u_out = propagate_asm(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
            else:
                u_out = propagate_blas(u_in, wavelength, z=z_eval, dx=dx, dy=dx, padding=2)
            t_elapsed = time.perf_counter() - t0

            i_out = np.abs(u_out)**2
            err_raw = relative_l2_error(i_out, i_ref)
            err_peak = relative_l2_error(i_out / np.max(i_out), i_ref / np.max(i_ref))

            metrics_rows.append({
                "experiment": "fixed_fov",
                "N": n,
                "dx_um": dx * 1e6,
                "fov_mm": fov * 1e3,
                "z_m": z_eval,
                "method": method,
                "runtime_s": round(t_elapsed, 4),
                "relative_error_raw_pct": round(err_raw * 100.0, 3),
                "relative_error_peak_pct": round(err_peak * 100.0, 3),
            })

    # Far-field aliasing comparison at z = 0.80 m
    n_ff = 512
    dx_ff = 5e-6
    grid_ff = create_grid(nx=n_ff, ny=n_ff, dx=dx_ff, dy=dx_ff)
    u_in_ff = circular_aperture(grid_ff, radius=radius)
    i_ref_ff = airy_pattern(grid_ff, wavelength, z=0.80, radius=radius)

    for method in ["fresnel", "asm", "blas"]:
        t0 = time.perf_counter()
        if method == "fresnel":
            u_out = propagate_fresnel(u_in_ff, wavelength, z=0.80, dx=dx_ff, dy=dx_ff, padding=2)
        elif method == "asm":
            u_out = propagate_asm(u_in_ff, wavelength, z=0.80, dx=dx_ff, dy=dx_ff, padding=2)
        else:
            u_out = propagate_blas(u_in_ff, wavelength, z=0.80, dx=dx_ff, dy=dx_ff, padding=2)
        t_elapsed = time.perf_counter() - t0

        i_out = np.abs(u_out)**2
        err_raw = relative_l2_error(i_out, i_ref_ff)
        err_peak = relative_l2_error(i_out / np.max(i_out), i_ref_ff / np.max(i_ref_ff))

        metrics_rows.append({
            "experiment": "far_field_aliasing",
            "N": n_ff,
            "dx_um": dx_ff * 1e6,
            "fov_mm": n_ff * dx_ff * 1e3,
            "z_m": 0.80,
            "method": method,
            "runtime_s": round(t_elapsed, 4),
            "relative_error_raw_pct": round(err_raw * 100.0, 3),
            "relative_error_peak_pct": round(err_peak * 100.0, 3),
        })

    return metrics_rows


def save_results(validations: Dict[str, Any], metrics: List[Dict[str, Any]]) -> None:
    """Save validation dictionary to JSON and benchmark metrics to CSV."""
    os.makedirs("results", exist_ok=True)

    with open("results/validation.json", "w", encoding="utf-8") as f:
        json.dump(validations, f, indent=2)

    import csv
    with open("results/metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    parser = argparse.ArgumentParser(description="Reproduce scalar diffraction experiments.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification with smaller grids.")
    parser.add_argument("--figure-only", action="store_true", help="Only generate publication figures 1-3.")
    parser.add_argument("--gif-only", action="store_true", help="Only generate animated loop GIFs.")
    parser.add_argument("--all", action="store_true", help="Run validation, benchmarks, and all figures and GIFs.")
    args = parser.parse_args()

    if args.figure_only:
        print("Generating Figures 1, 2, and 3...")
        generate_fig01_aperture_diffraction()
        generate_fig02_methods_errors()
        generate_fig03_results_profiles()
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

    print("Benchmarking propagation methods and sampling convergence...")
    metrics = run_benchmark_metrics()
    save_results(validations, metrics)
    print("Metrics written to results/metrics.csv and results/validation.json")

    print("Rendering Figures 1, 2, and 3...")
    generate_fig01_aperture_diffraction()
    generate_fig02_methods_errors()
    generate_fig03_results_profiles()

    print("Rendering animated GIFs...")
    n1 = generate_primary_loop_gif()
    print(f"Primary GIF saved ({n1} frames).")
    n2 = generate_error_loop_gif()
    print(f"Error GIF saved ({n2} frames).")
    print("All artifacts generated successfully.")


if __name__ == "__main__":
    main()
