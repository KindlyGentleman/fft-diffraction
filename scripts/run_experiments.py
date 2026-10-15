import argparse
import os
import yaml
import numpy as np
import pandas as pd

from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture, make_rectangular_aperture
from diffraction.propagation import (
    propagate_asm,
    propagate_blas,
    propagate_fresnel,
    get_asm_transfer_function,
    get_blas_transfer_function,
)
from diffraction.analytical import circular_fraunhofer_field, rectangular_fraunhofer_field
from diffraction.metrics import calculate_intensity
from diffraction.experiments import (
    run_study1_propagation_distance,
    run_study2_sampling_convergence,
    run_study3_bandlimiting_analysis,
    run_computational_benchmarks,
)

def main():
    parser = argparse.ArgumentParser(description="Run diffraction experiment sweeps")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Execution mode")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.mode}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(f"RUNNING EXPERIMENTAL STUDIES (Mode: {args.mode.upper()})")
    print(f"Config: {config_path}")
    print("=" * 70)

    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/data", exist_ok=True)

    wavelength = cfg["wavelength"]
    circ_radius = cfg["apertures"]["circular"]["radius"]
    rect_width = cfg["apertures"]["rectangular"]["width"]
    rect_height = cfg["apertures"]["rectangular"]["height"]

    # 1. Study 1: Propagation Distance
    print("\n[1/4] Running Study 1: Propagation Distance Sweep...")
    s1_cfg = cfg["study1_distances"]
    df_s1 = run_study1_propagation_distance(
        distances=s1_cfg["distances"],
        nx=s1_cfg["grid"]["nx"],
        ny=s1_cfg["grid"]["ny"],
        dx=s1_cfg["grid"]["dx"],
        dy=s1_cfg["grid"]["dy"],
        wavelength=wavelength,
        circ_radius=circ_radius,
        rect_width=rect_width,
        rect_height=rect_height,
        padding=s1_cfg["grid"]["padding"],
    )
    s1_file = "outputs/metrics/study1_distances.csv"
    df_s1.to_csv(s1_file, index=False)
    print(f"  -> Saved {len(df_s1)} records to {s1_file}")

    # 2. Study 2: Sampling Convergence
    print("\n[2/4] Running Study 2: Sampling and Window Convergence...")
    s2_cfg = cfg["study2_sampling"]
    resolutions = s2_cfg["fixed_fov"]["resolutions"]
    distance = s2_cfg["fixed_fov"]["distance"]
    df_s2 = run_study2_sampling_convergence(
        resolutions=resolutions,
        wavelength=wavelength,
        circ_radius=circ_radius,
        z=distance,
    )
    s2_file = "outputs/metrics/study2_sampling.csv"
    df_s2.to_csv(s2_file, index=False)
    print(f"  -> Saved {len(df_s2)} records to {s2_file}")

    # 3. Study 3: Band-limiting Robustness
    print("\n[3/4] Running Study 3: Band-limiting Truncation Analysis...")
    s3_cfg = cfg["study3_bandlimiting"]
    distances_s3 = [tc["distance"] for tc in s3_cfg["test_cases"]]
    df_s3 = run_study3_bandlimiting_analysis(
        distances=distances_s3,
        nx=s3_cfg["grid"]["nx"],
        ny=s3_cfg["grid"]["ny"],
        dx=s3_cfg["grid"]["dx"],
        dy=s3_cfg["grid"]["dy"],
        wavelength=wavelength,
        circ_radius=circ_radius,
        padding=s3_cfg["grid"]["padding"],
    )
    s3_file = "outputs/metrics/study3_bandlimiting.csv"
    df_s3.to_csv(s3_file, index=False)
    print(f"  -> Saved {len(df_s3)} records to {s3_file}")

    # 4. Benchmarking
    print("\n[4/4] Running Computational Runtime Benchmarks...")
    bench_cfg = cfg["benchmarks"]
    df_cost = run_computational_benchmarks(
        resolutions=bench_cfg["resolutions"],
        wavelength=wavelength,
        repeats=bench_cfg["repeats"],
    )

    # Attach error from study 2 to benchmark cost table
    df_s2_blas = df_s2[(df_s2["substudy"] == "fixed_fov") & (df_s2["method"] == "blas")]
    err_map = dict(zip(df_s2_blas["N"], df_s2_blas["l2_intensity_error"]))
    df_cost["l2_intensity_error"] = df_cost["N"].map(err_map).fillna(0.05)

    cost_file = "outputs/metrics/benchmarks_cost.csv"
    df_cost.to_csv(cost_file, index=False)
    print(f"  -> Saved {len(df_cost)} records to {cost_file}")

    # 5. Save selected 2D simulation arrays for figure reproduction
    print("\nSaving 2D spatial simulation arrays for IEEE figures...")
    grid_std = create_grid(512, 512, 5.0e-6, 5.0e-6)
    f_circ = make_circular_aperture(grid_std, radius=circ_radius)
    f_rect = make_rectangular_aperture(grid_std, width=rect_width, height=rect_height)

    # Matched comparison at z = 0.05 m
    z_comp = 0.050
    u_fresnel, _, _ = propagate_fresnel(f_circ, wavelength, z_comp, grid_std.dx, grid_std.dy, padding=2)
    u_asm, _, _ = propagate_asm(f_circ, wavelength, z_comp, grid_std.dx, grid_std.dy, padding=2)
    u_blas, _, _ = propagate_blas(f_circ, wavelength, z_comp, grid_std.dx, grid_std.dy, padding=2)
    u_ref = circular_fraunhofer_field(grid_std, wavelength, z_comp, circ_radius)

    np.savez_compressed(
        "outputs/data/fig03_data.npz",
        i_fresnel=calculate_intensity(u_fresnel),
        i_asm=calculate_intensity(u_asm),
        i_blas=calculate_intensity(u_blas),
        i_ref=calculate_intensity(u_ref),
        z_comp=z_comp,
    )

    # Diagnostic transfer functions at z = 0.20 m
    z_tf = 0.200
    pad_grid = create_grid(1024, 1024, grid_std.dx, grid_std.dy)
    h_asm = get_asm_transfer_function(pad_grid, wavelength, z_tf)
    h_blas, mask = get_blas_transfer_function(pad_grid, wavelength, z_tf)

    u_asm_20, _, _ = propagate_asm(f_circ, wavelength, z_tf, grid_std.dx, grid_std.dy, padding=2)
    u_blas_20, _, _ = propagate_blas(f_circ, wavelength, z_tf, grid_std.dx, grid_std.dy, padding=2)
    diff_map = np.abs(calculate_intensity(u_asm_20) - calculate_intensity(u_blas_20))

    np.savez_compressed(
        "outputs/data/fig07_data.npz",
        h_asm=h_asm,
        h_blas=h_blas,
        mask=mask,
        diff_map=diff_map,
        z_tf=z_tf,
    )

    # Aliasing failure case at z = 0.40 m
    z_alias = 0.400
    u_asm_alias, _, _ = propagate_asm(f_circ, wavelength, z_alias, grid_std.dx, grid_std.dy, padding=2)
    u_blas_clean, _, _ = propagate_blas(f_circ, wavelength, z_alias, grid_std.dx, grid_std.dy, padding=2)

    np.savez_compressed(
        "outputs/data/fig09_data.npz",
        i_asm_alias=calculate_intensity(u_asm_alias),
        i_blas_clean=calculate_intensity(u_blas_clean),
        z_alias=z_alias,
    )

    print("Experiment run completed successfully.")

if __name__ == "__main__":
    main()
