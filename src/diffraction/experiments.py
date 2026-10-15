import os
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture, make_rectangular_aperture
from diffraction.propagation import propagate_asm, propagate_blas, propagate_fresnel
from diffraction.analytical import (
    circular_fraunhofer_field,
    rectangular_fraunhofer_field,
    circular_fraunhofer_intensity,
    rectangular_fraunhofer_intensity,
)
from diffraction.metrics import calculate_metrics, calculate_intensity, calculate_energy
from diffraction.bandlimit import compute_bandlimit_frequencies, make_bandlimit_mask, analyze_spectral_support
from diffraction.utils import BenchmarkTimer

def run_study1_propagation_distance(
    distances: List[float],
    nx: int = 512,
    ny: int = 512,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    circ_radius: float = 1.5e-4,
    rect_width: float = 3.0e-4,
    rect_height: float = 4.5e-4,
    padding: int = 2,
) -> pd.DataFrame:
    """Study 1: Evaluate propagation methods across propagation distance z."""
    grid = create_grid(nx, ny, dx, dy)
    field_circ = make_circular_aperture(grid, radius=circ_radius)
    field_rect = make_rectangular_aperture(grid, width=rect_width, height=rect_height)

    records = []

    for z in distances:
        fresnel_num_circ = (circ_radius**2) / (wavelength * z) if z > 0 else np.nan

        u_ana_circ = circular_fraunhofer_field(grid, wavelength, z, circ_radius)
        u_ana_rect = rectangular_fraunhofer_field(grid, wavelength, z, rect_width, rect_height)

        for ap_name, field_in, u_ref in [
            ("circular", field_circ, u_ana_circ),
            ("rectangular", field_rect, u_ana_rect),
        ]:
            for method_name, prop_fn in [
                ("fresnel", propagate_fresnel),
                ("asm", propagate_asm),
                ("blas", propagate_blas),
            ]:
                with BenchmarkTimer() as timer:
                    u_out, _, H = prop_fn(
                        field=field_in,
                        wavelength=wavelength,
                        z=z,
                        dx=dx,
                        dy=dy,
                        padding=padding,
                        return_transfer=True,
                    )

                metrics = calculate_metrics(u_out, u_ref, dx, dy)

                if method_name == "blas":
                    pad_nx = nx * padding
                    pad_ny = ny * padding
                    pad_grid = create_grid(pad_nx, pad_ny, dx, dy)
                    mask = make_bandlimit_mask(pad_grid, wavelength, z)
                    retained_frac = float(np.count_nonzero(mask) / mask.size)
                else:
                    retained_frac = 1.0

                records.append({
                    "study": "distance_sweep",
                    "aperture": ap_name,
                    "method": method_name,
                    "z_m": float(z),
                    "fresnel_number": float(fresnel_num_circ),
                    "runtime_s": float(timer.elapsed),
                    "l2_intensity_error": float(metrics["l2_intensity_error"]),
                    "l2_complex_error": float(metrics["l2_complex_error"]),
                    "l2_phase_invariant_error": float(metrics["l2_phase_invariant_error"]),
                    "energy_discrepancy": float(metrics["energy_discrepancy"]),
                    "retained_spectral_fraction": float(retained_frac),
                })

    return pd.DataFrame(records)

def run_study2_sampling_convergence(
    resolutions: List[int],
    wavelength: float = 5.32e-7,
    circ_radius: float = 1.5e-4,
    z: float = 0.500,
) -> pd.DataFrame:
    """Study 2: Investigate convergence under fixed FOV vs fixed pitch."""
    records = []

    # Experiment 2A: Fixed Field of View
    fixed_fov = 2.56e-3
    ref_n = 2048
    ref_dx = fixed_fov / ref_n
    ref_grid = create_grid(ref_n, ref_n, ref_dx, ref_dx)
    ref_field_in = make_circular_aperture(ref_grid, radius=circ_radius)
    u_ref_fov, _, _ = propagate_blas(ref_field_in, wavelength, z, ref_dx, ref_dx, padding=2)

    for n in resolutions:
        dx = fixed_fov / n
        grid = create_grid(n, n, dx, dx)
        field_in = make_circular_aperture(grid, radius=circ_radius)

        for method_name, prop_fn in [
            ("fresnel", propagate_fresnel),
            ("asm", propagate_asm),
            ("blas", propagate_blas),
        ]:
            u_out, _, _ = prop_fn(field_in, wavelength, z, dx, dx, padding=2)
            u_ana = circular_fraunhofer_field(grid, wavelength, z, circ_radius)
            metrics = calculate_metrics(u_out, u_ana, dx, dx)

            records.append({
                "substudy": "fixed_fov",
                "N": int(n),
                "dx_um": float(dx * 1e6),
                "fov_mm": float(fixed_fov * 1e3),
                "method": method_name,
                "z_m": float(z),
                "l2_intensity_error": float(metrics["l2_intensity_error"]),
                "l2_complex_error": float(metrics["l2_complex_error"]),
                "energy_discrepancy": float(metrics["energy_discrepancy"]),
            })

    # Experiment 2B: Fixed Pitch
    fixed_dx = 5.0e-6
    for n in resolutions:
        fov = n * fixed_dx
        grid = create_grid(n, n, fixed_dx, fixed_dx)
        field_in = make_circular_aperture(grid, radius=circ_radius)

        for method_name, prop_fn in [
            ("fresnel", propagate_fresnel),
            ("asm", propagate_asm),
            ("blas", propagate_blas),
        ]:
            u_out, _, _ = prop_fn(field_in, wavelength, z, fixed_dx, fixed_dx, padding=2)
            u_ana = circular_fraunhofer_field(grid, wavelength, z, circ_radius)
            metrics = calculate_metrics(u_out, u_ana, fixed_dx, fixed_dx)

            records.append({
                "substudy": "fixed_pitch",
                "N": int(n),
                "dx_um": float(fixed_dx * 1e6),
                "fov_mm": float(fov * 1e3),
                "method": method_name,
                "z_m": float(z),
                "l2_intensity_error": float(metrics["l2_intensity_error"]),
                "l2_complex_error": float(metrics["l2_complex_error"]),
                "energy_discrepancy": float(metrics["energy_discrepancy"]),
            })

    return pd.DataFrame(records)

def run_study3_bandlimiting_analysis(
    distances: List[float],
    nx: int = 512,
    ny: int = 512,
    dx: float = 5.0e-6,
    dy: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    circ_radius: float = 1.5e-4,
    padding: int = 2,
) -> pd.DataFrame:
    """Study 3: Robustness and spectral energy truncation of band-limiting."""
    grid = create_grid(nx, ny, dx, dy)
    field_in = make_circular_aperture(grid, radius=circ_radius)

    records = []

    for z in distances:
        u_asm, _, _ = propagate_asm(field_in, wavelength, z, dx, dy, padding=padding)
        u_blas, _, _ = propagate_blas(field_in, wavelength, z, dx, dy, padding=padding)
        u_fresnel, _, _ = propagate_fresnel(field_in, wavelength, z, dx, dy, padding=padding)

        pad_grid = create_grid(nx * padding, ny * padding, dx, dy)
        mask = make_bandlimit_mask(pad_grid, wavelength, z)
        fx_limit, fy_limit = compute_bandlimit_frequencies(wavelength, z, pad_grid.Sx, pad_grid.Sy)

        u_ana = circular_fraunhofer_field(grid, wavelength, z, circ_radius)
        met_asm = calculate_metrics(u_asm, u_ana, dx, dy)
        met_blas = calculate_metrics(u_blas, u_ana, dx, dy)
        met_fresnel = calculate_metrics(u_fresnel, u_ana, dx, dy)

        e_asm = calculate_energy(u_asm, dx, dy)
        e_blas = calculate_energy(u_blas, dx, dy)
        energy_loss_pct = ((e_asm - e_blas) / e_asm * 100.0) if e_asm > 0 else 0.0

        records.append({
            "z_m": float(z),
            "fx_limit_per_mm": float(fx_limit * 1e-3),
            "retained_spectral_fraction": float(np.count_nonzero(mask) / mask.size),
            "energy_loss_pct": float(energy_loss_pct),
            "l2_err_asm": float(met_asm["l2_intensity_error"]),
            "l2_err_blas": float(met_blas["l2_intensity_error"]),
            "l2_err_fresnel": float(met_fresnel["l2_intensity_error"]),
            "blas_improvement_over_asm": float(met_asm["l2_intensity_error"] - met_blas["l2_intensity_error"]),
        })

    return pd.DataFrame(records)

def run_computational_benchmarks(
    resolutions: List[int],
    dx: float = 5.0e-6,
    wavelength: float = 5.32e-7,
    z: float = 0.050,
    repeats: int = 3,
    padding: int = 2,
) -> pd.DataFrame:
    """Benchmark pure propagation execution times across methods."""
    records = []

    for n in resolutions:
        grid = create_grid(n, n, dx, dx)
        field_in = make_circular_aperture(grid, radius=1.5e-4)

        for method_name, prop_fn in [
            ("fresnel", propagate_fresnel),
            ("asm", propagate_asm),
            ("blas", propagate_blas),
        ]:
            # Warm up cache and JIT/FFT planning
            prop_fn(field_in, wavelength, z, dx, dx, padding=padding)

            timings = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                prop_fn(field_in, wavelength, z, dx, dx, padding=padding)
                t1 = time.perf_counter()
                timings.append(t1 - t0)

            median_time = float(np.median(timings))
            std_time = float(np.std(timings))

            records.append({
                "N": int(n),
                "method": method_name,
                "median_runtime_s": median_time,
                "std_runtime_s": std_time,
                "repeats": repeats,
            })

    return pd.DataFrame(records)
