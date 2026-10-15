import os
import numpy as np
import pandas as pd

from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture, make_rectangular_aperture
from diffraction.propagation import propagate_blas
from diffraction.analytical import (
    circular_fraunhofer_intensity,
    rectangular_fraunhofer_intensity,
    airy_first_zero_radius,
)
from diffraction.metrics import calculate_intensity
from diffraction.utils import radial_average
from diffraction.plotting import (
    plot_fig01_geometry,
    plot_fig02_apertures,
    plot_fig03_method_comparison,
    plot_fig04_analytical_validation,
    plot_fig05_sampling_convergence,
    plot_fig06_distance_study,
    plot_fig07_bandlimiting_analysis,
    plot_fig08_cost_accuracy,
    plot_fig09_failure_case,
)

def main():
    print("=" * 70)
    print("GENERATING IEEE-STYLE PUBLICATION FIGURES (SciencePlots)")
    print("=" * 70)

    fig_dir = os.path.join("outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    grid = create_grid(512, 512, 5.0e-6, 5.0e-6)
    circ_radius = 1.5e-4
    rect_width = 3.0e-4
    rect_height = 4.5e-4
    wavelength = 5.32e-7

    field_circ = make_circular_aperture(grid, radius=circ_radius)
    field_rect = make_rectangular_aperture(grid, width=rect_width, height=rect_height)

    # Figure 01: Geometry Schematic
    print("[1/9] Generating Fig 01: Diffraction Geometry...")
    plot_fig01_geometry(fig_dir)

    # Figure 02: Input Apertures
    print("[2/9] Generating Fig 02: Input Apertures...")
    plot_fig02_apertures(grid, field_circ, field_rect, fig_dir)

    # Figure 03: Method Comparison
    print("[3/9] Generating Fig 03: Three-Method Comparison...")
    fig03_data_file = "outputs/data/fig03_data.npz"
    if os.path.exists(fig03_data_file):
        data03 = np.load(fig03_data_file)
        plot_fig03_method_comparison(
            grid=grid,
            i_fresnel=data03["i_fresnel"],
            i_asm=data03["i_asm"],
            i_blas=data03["i_blas"],
            i_ref=data03["i_ref"],
            z_val=float(data03["z_comp"]),
            output_dir=fig_dir,
        )
    else:
        print("  Notice: fig03_data.npz not found, computing on the fly...")
        z_comp = 0.050
        u_fresnel, _, _ = propagate_blas(field_circ, wavelength, z_comp, grid.dx, grid.dy, padding=2)
        u_asm, _, _ = propagate_blas(field_circ, wavelength, z_comp, grid.dx, grid.dy, padding=2)
        u_blas, _, _ = propagate_blas(field_circ, wavelength, z_comp, grid.dx, grid.dy, padding=2)
        i_ref = circular_fraunhofer_intensity(grid, wavelength, z_comp, circ_radius)
        plot_fig03_method_comparison(
            grid=grid,
            i_fresnel=calculate_intensity(u_fresnel),
            i_asm=calculate_intensity(u_asm),
            i_blas=calculate_intensity(u_blas),
            i_ref=i_ref,
            z_val=z_comp,
            output_dir=fig_dir,
        )

    # Figure 04: Analytical Validation
    print("[4/9] Generating Fig 04: Analytical Validation...")
    z_val_d = 0.500
    u_blas_d, _, _ = propagate_blas(field_circ, wavelength, z_val_d, grid.dx, grid.dy, padding=2)
    i_num_d = calculate_intensity(u_blas_d)
    i_ana_d = circular_fraunhofer_intensity(grid, wavelength, z_val_d, circ_radius)

    r_num, prof_num = radial_average(i_num_d, grid.x, grid.y, num_bins=150)
    r_ana, prof_ana = radial_average(i_ana_d, grid.x, grid.y, num_bins=150)
    r1_theory = airy_first_zero_radius(wavelength, z_val_d, circ_radius)

    u_blas_rect, _, _ = propagate_blas(field_rect, wavelength, z_val_d, grid.dx, grid.dy, padding=2)
    i_rect_num = calculate_intensity(u_blas_rect)
    i_rect_ana = rectangular_fraunhofer_intensity(grid, wavelength, z_val_d, rect_width, rect_height)

    cy = grid.ny // 2
    slice_x_num = i_rect_num[cy, :]
    slice_x_ana = i_rect_ana[cy, :]

    plot_fig04_analytical_validation(
        r_coords=r_num,
        prof_num=prof_num,
        prof_ana=prof_ana,
        r1_theory=r1_theory,
        x_coords=grid.x,
        slice_x_num=slice_x_num,
        slice_x_ana=slice_x_ana,
        output_dir=fig_dir,
    )

    # Figure 05: Sampling Convergence
    print("[5/9] Generating Fig 05: Sampling Convergence...")
    s2_file = "outputs/metrics/study2_sampling.csv"
    if os.path.exists(s2_file):
        df_sampling = pd.read_csv(s2_file)
        plot_fig05_sampling_convergence(df_sampling, fig_dir)
    else:
        print("  Warning: study2_sampling.csv missing, run run_experiments.py first.")

    # Figure 06: Distance Study
    print("[6/9] Generating Fig 06: Distance Study...")
    s1_file = "outputs/metrics/study1_distances.csv"
    if os.path.exists(s1_file):
        df_dist = pd.read_csv(s1_file)
        plot_fig06_distance_study(df_dist, fig_dir)
    else:
        print("  Warning: study1_distances.csv missing, run run_experiments.py first.")

    # Figure 07: Band-limiting Analysis
    print("[7/9] Generating Fig 07: Band-limiting Diagnostics...")
    fig07_data_file = "outputs/data/fig07_data.npz"
    if os.path.exists(fig07_data_file):
        data07 = np.load(fig07_data_file)
        pad_grid = create_grid(1024, 1024, grid.dx, grid.dy)
        plot_fig07_bandlimiting_analysis(
            grid_pad=pad_grid,
            h_asm=data07["h_asm"],
            h_blas=data07["h_blas"],
            mask=data07["mask"],
            diff_map=data07["diff_map"],
            z_val=float(data07["z_tf"]),
            output_dir=fig_dir,
        )
    else:
        print("  Warning: fig07_data.npz missing.")

    # Figure 08: Cost vs Accuracy
    print("[8/9] Generating Fig 08: Computational Cost vs Accuracy...")
    cost_file = "outputs/metrics/benchmarks_cost.csv"
    if os.path.exists(cost_file):
        df_cost = pd.read_csv(cost_file)
        plot_fig08_cost_accuracy(df_cost, fig_dir)
    else:
        print("  Warning: benchmarks_cost.csv missing.")

    # Figure 09: Failure Case
    print("[9/9] Generating Fig 09: Numerical Failure Case...")
    fig09_data_file = "outputs/data/fig09_data.npz"
    if os.path.exists(fig09_data_file):
        data09 = np.load(fig09_data_file)
        plot_fig09_failure_case(
            grid=grid,
            i_asm_aliased=data09["i_asm_alias"],
            i_blas_clean=data09["i_blas_clean"],
            z_val=float(data09["z_alias"]),
            output_dir=fig_dir,
        )

    print("\nAll IEEE figures generated in PDF, SVG, and PNG format.")

if __name__ == "__main__":
    main()
