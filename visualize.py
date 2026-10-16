"""Scientific visualization module for FFT scalar diffraction experiments.

Generates:
1. Fig 1: 2x2 Input apertures aligned above corresponding diffraction fields.
2. Fig 2: 2x4 Fair comparison of Fresnel, ASM, BL-ASM, and SAS with error maps.
3. Fig 3: 2x2 Analytical Fraunhofer validation with quantitative cross-sections.
4. Fig 4: 2x3 Propagation method-selection map across sampling and distance.
5. Fig 5: 2x3 Gaussian beam focusing through a thin lens vs ABCD prediction.
6. Fig 6: 2x3 Optical vortex propagation and topological charge conservation.
7. Primary GIF: Animated free-space propagation loop.
8. Error GIF: Animated ASM vs SAS error evolution loop.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image

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
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    radial_profile,
    find_first_radial_minimum,
    relative_l2_error,
    compute_shared_region_l2_error,
    compare_on_common_grid,
    sas_output_scaling,
    thin_lens_phase,
    spiral_phase,
    topological_charge,
    gaussian_beam_abcd,
    check_sas_validity,
    measure_beam_radius,
)

# Apply IEEE publication style
try:
    import scienceplots
    plt.style.use(["science", "ieee", "no-latex"])
except Exception:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.titlesize": 9,
        "mathtext.fontset": "dejavuserif",
    })

COLORS = {
    "fresnel": "#d95f02",
    "asm": "#386cb0",
    "blas": "#1b9e77",
    "sas": "#7570b3",
    "ref": "#222222",
}


def generate_fig01_aperture_diffraction(
    output_png: str = "assets/fig01_aperture_diffraction.png",
    output_pdf: str = "assets/fig01_aperture_diffraction.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 1: 2x2 grid of input apertures aligned above diffraction fields."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    z_prop = 0.50
    radius = 1.5e-4
    w_rect, h_rect = 3.0e-4, 4.5e-4

    grid = create_grid(nx, ny, dx, dy)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    u_circ_in = circular_aperture(grid, radius=radius)
    u_rect_in = rectangular_aperture(grid, width=w_rect, height=h_rect)

    u_circ_out = propagate_blas(u_circ_in, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)
    u_rect_out = propagate_blas(u_rect_in, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)

    i_circ = np.abs(u_circ_out)**2
    i_rect = np.abs(u_rect_out)**2

    i_circ_norm = i_circ / np.max(i_circ)
    i_rect_norm = i_rect / np.max(i_rect)

    fig = plt.figure(figsize=(7.16, 6.2), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # Panel (a): Input Circular Aperture
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(np.abs(u_circ_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_a.set_title("(a) Circular Aperture ($2a = 0.30$ mm)", fontsize=8)
    ax_a.set_xlabel("$x$ [mm]")
    ax_a.set_ylabel("$y$ [mm]")

    # Panel (b): Input Rectangular Aperture
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(np.abs(u_rect_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_b.set_title("(b) Rectangular Aperture ($0.30 \\times 0.45$ mm)", fontsize=8)
    ax_b.set_xlabel("$x$ [mm]")
    ax_b.set_ylabel("$y$ [mm]")

    # Panel (c): Circular Aperture Diffraction Result
    ax_c = fig.add_subplot(gs[1, 0])
    im_c = ax_c.imshow(i_circ_norm, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_c.set_title("(c) Circular Diffraction ($z = 0.50$ m)", fontsize=8)
    ax_c.set_xlabel("$x$ [mm]")
    ax_c.set_ylabel("$y$ [mm]")

    # Panel (d): Rectangular Aperture Diffraction Result
    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(i_rect_norm, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_d.set_title("(d) Rectangular Diffraction ($z = 0.50$ m)", fontsize=8)
    ax_d.set_xlabel("$x$ [mm]")
    ax_d.set_ylabel("$y$ [mm]")

    cbar = fig.colorbar(im_d, ax=[ax_c, ax_d], orientation="horizontal", fraction=0.045, pad=0.18, aspect=35)
    cbar.set_label("Normalized Intensity $I / I_{\\max}$", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    fig.suptitle("Input Aperture Transmission and Propagated Scalar Diffraction Fields ($\\lambda = 532$ nm)", fontsize=9, y=0.98)

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig02_methods_errors(
    output_png: str = "assets/fig02_methods_errors.png",
    output_pdf: str = "assets/fig02_methods_errors.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 2: Fair 2x4 comparison of Fresnel, ASM, BL-ASM, and SAS with error maps."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4
    z_eval = 0.80

    grid = create_grid(nx, ny, dx, dy)
    extent_fixed = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    u_in = circular_aperture(grid, radius=radius)
    u_fres = propagate_fresnel(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_asm = propagate_asm(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_blas = propagate_blas(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_sas, grid_sas = propagate_sas(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)

    extent_sas = [grid_sas.x[0] * 1e3, grid_sas.x[-1] * 1e3, grid_sas.y[0] * 1e3, grid_sas.y[-1] * 1e3]

    i_fres = np.abs(u_fres)**2
    i_asm = np.abs(u_asm)**2
    i_blas = np.abs(u_blas)**2
    i_sas = np.abs(u_sas)**2

    i_ref_fixed = airy_pattern(grid, wavelength, z=z_eval, radius=radius)
    i_ref_sas = airy_pattern(grid_sas, wavelength, z=z_eval, radius=radius)

    max_ref_fixed = np.max(i_ref_fixed)
    max_ref_sas = np.max(i_ref_sas)

    disp_fres = i_fres / max_ref_fixed
    disp_asm = i_asm / max_ref_fixed
    disp_blas = i_blas / max_ref_fixed
    disp_sas = i_sas / max_ref_sas

    err_fres = np.abs(disp_fres - i_ref_fixed / max_ref_fixed)
    err_asm = np.abs(disp_asm - i_ref_fixed / max_ref_fixed)
    err_blas = np.abs(disp_blas - i_ref_fixed / max_ref_fixed)
    err_sas = np.abs(disp_sas - i_ref_sas / max_ref_sas)

    # Fair shared physical domain evaluation [-1.28, 1.28] mm
    x_max = grid.Sx / 2.0
    l2_fres_shared = relative_l2_error(i_fres, i_ref_fixed) * 100.0
    l2_asm_shared = relative_l2_error(i_asm, i_ref_fixed) * 100.0
    l2_blas_shared = relative_l2_error(i_blas, i_ref_fixed) * 100.0
    l2_sas_shared = compute_shared_region_l2_error(i_sas, grid_sas, i_ref_sas, x_max=x_max, y_max=x_max) * 100.0
    l2_sas_full = relative_l2_error(i_sas, i_ref_sas) * 100.0

    fig = plt.figure(figsize=(7.16, 4.4), dpi=dpi)
    gs = GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.30)

    vmax_int = 1.05
    vmax_err = 0.20

    # Row 1: Intensity fields
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(disp_fres, extent=extent_fixed, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_a.set_title("(a) Fresnel (Fixed)", fontsize=7.5)
    ax_a.set_ylabel("$y$ [mm]")
    ax_a.tick_params(labelsize=6)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(disp_asm, extent=extent_fixed, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_b.set_title("(b) Standard ASM (Fixed)", fontsize=7.5)
    ax_b.tick_params(labelsize=6)

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(disp_blas, extent=extent_fixed, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_c.set_title("(c) Band-Limited ASM", fontsize=7.5)
    ax_c.tick_params(labelsize=6)

    ax_d = fig.add_subplot(gs[0, 3])
    im_d = ax_d.imshow(disp_sas, extent=extent_sas, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_d.set_title("(d) Scalable AS (Scaled)", fontsize=7.5)
    ax_d.tick_params(labelsize=6)
    # Highlight the shared region on the SAS scaled panel
    rect_d = Rectangle((-x_max * 1e3, -x_max * 1e3), 2 * x_max * 1e3, 2 * x_max * 1e3,
                       fill=False, edgecolor="#e41a1c", ls="--", lw=0.8)
    ax_d.add_patch(rect_d)
    ax_d.text(0.04, 0.05, "Shared FOV", transform=ax_d.transAxes, fontsize=5.5, color="#e41a1c",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, lw=0))

    # Row 2: Error maps
    ax_e = fig.add_subplot(gs[1, 0])
    im_e = ax_e.imshow(err_fres, extent=extent_fixed, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_e.set_title("(e) Fresnel Error", fontsize=7.5)
    ax_e.set_xlabel("$x$ [mm]")
    ax_e.set_ylabel("$y$ [mm]")
    ax_e.tick_params(labelsize=6)
    ax_e.text(0.04, 0.80, f"Shared $L_2$: {l2_fres_shared:.1f}%\nFOV: 2.56 mm (fixed)",
              transform=ax_e.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.imshow(err_asm, extent=extent_fixed, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_f.set_title("(f) ASM Error (Aliasing)", fontsize=7.5)
    ax_f.set_xlabel("$x$ [mm]")
    ax_f.tick_params(labelsize=6)
    ax_f.text(0.04, 0.80, f"Shared $L_2$: {l2_asm_shared:.1f}%\nFOV: 2.56 mm (fixed)",
              transform=ax_f.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.imshow(err_blas, extent=extent_fixed, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_g.set_title("(g) BL-ASM Error", fontsize=7.5)
    ax_g.set_xlabel("$x$ [mm]")
    ax_g.tick_params(labelsize=6)
    ax_g.text(0.04, 0.80, f"Shared $L_2$: {l2_blas_shared:.1f}%\nFOV: 2.56 mm (fixed)",
              transform=ax_g.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    ax_h = fig.add_subplot(gs[1, 3])
    im_h = ax_h.imshow(err_sas, extent=extent_sas, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_h.set_title("(h) SAS Error", fontsize=7.5)
    ax_h.set_xlabel("$x$ [mm]")
    ax_h.tick_params(labelsize=6)
    rect_h = Rectangle((-x_max * 1e3, -x_max * 1e3), 2 * x_max * 1e3, 2 * x_max * 1e3,
                       fill=False, edgecolor="#e41a1c", ls="--", lw=0.8)
    ax_h.add_patch(rect_h)
    ax_h.text(0.04, 0.80, f"Shared: {l2_sas_shared:.1f}% | Full: {l2_sas_full:.1f}%\nFOV: 42.56 mm (scaled)",
              transform=ax_h.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    # Shared colorbars
    cbar_top = fig.colorbar(im_d, ax=[ax_a, ax_b, ax_c, ax_d], orientation="vertical", fraction=0.015, pad=0.02)
    cbar_top.set_label("$I / I_{\\mathrm{ref},\\max}$", fontsize=6.5)
    cbar_top.ax.tick_params(labelsize=5.5)

    cbar_bot = fig.colorbar(im_h, ax=[ax_e, ax_f, ax_g, ax_h], orientation="vertical", fraction=0.015, pad=0.02)
    cbar_bot.set_label("$|I - I_{\\mathrm{ref}}| / I_{\\mathrm{ref},\\max}$", fontsize=6.5)
    cbar_bot.ax.tick_params(labelsize=5.5)

    fig.suptitle("Fair Comparison of Scalar Propagation Methods ($\\lambda = 532$ nm, $z = 0.80$ m)", fontsize=8.5, y=0.98)

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig03_results_profiles(
    output_png: str = "assets/fig03_results_profiles.png",
    output_pdf: str = "assets/fig03_results_profiles.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 3: Analytical Fraunhofer validation with quantitative profiles."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    z_eval = 0.50
    radius = 1.5e-4
    w_rect, h_rect = 3.0e-4, 4.5e-4

    grid_in = create_grid(nx, ny, dx, dy)
    u_circ_in = circular_aperture(grid_in, radius=radius)
    u_rect_in = rectangular_aperture(grid_in, width=w_rect, height=h_rect)

    u_circ_sas, grid_circ = propagate_sas(u_circ_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_rect_sas, grid_rect = propagate_sas(u_rect_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_circ_blas = propagate_blas(u_circ_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)

    extent_circ = [grid_circ.x[0] * 1e3, grid_circ.x[-1] * 1e3, grid_circ.y[0] * 1e3, grid_circ.y[-1] * 1e3]
    extent_rect = [grid_rect.x[0] * 1e3, grid_rect.x[-1] * 1e3, grid_rect.y[0] * 1e3, grid_rect.y[-1] * 1e3]

    i_circ = np.abs(u_circ_sas)**2
    i_rect = np.abs(u_rect_sas)**2
    i_circ_blas = np.abs(u_circ_blas)**2

    i_circ_ana = airy_pattern(grid_circ, wavelength, z=z_eval, radius=radius)
    i_rect_ana = sinc_pattern(grid_rect, wavelength, z=z_eval, width=w_rect, height=h_rect)

    r_sas, prof_circ_sas = radial_profile(i_circ, grid_circ.x, grid_circ.y, num_bins=300)
    r_ana, prof_circ_ana = radial_profile(i_circ_ana, grid_circ.x, grid_circ.y, num_bins=300)
    r_blas, prof_circ_blas = radial_profile(i_circ_blas, grid_in.x, grid_in.y, num_bins=150)

    r1_theory = airy_first_zero(wavelength, z_eval, radius)
    r1_meas = find_first_radial_minimum(r_sas, prof_circ_sas, search_range=(0.7 * r1_theory, 1.4 * r1_theory))
    r1_err = abs(r1_meas - r1_theory) / r1_theory * 100.0

    p_sas_n = prof_circ_sas / np.max(prof_circ_sas)
    p_ana_n = prof_circ_ana / np.max(prof_circ_ana)
    p_blas_n = prof_circ_blas / np.max(prof_circ_blas)
    prof_l2_err = np.linalg.norm(p_sas_n - p_ana_n) / np.linalg.norm(p_ana_n) * 100.0

    cx, cy = nx // 2, ny // 2
    x_num = i_rect[cy, :] / np.max(i_rect[cy, :])
    x_ana = i_rect_ana[cy, :] / np.max(i_rect_ana[cy, :])
    y_num = i_rect[:, cx] / np.max(i_rect[:, cx])
    y_ana = i_rect_ana[:, cx] / np.max(i_rect_ana[:, cx])

    err_x = np.linalg.norm(x_num - x_ana) / np.linalg.norm(x_ana) * 100.0
    err_y = np.linalg.norm(y_num - y_ana) / np.linalg.norm(y_ana) * 100.0

    fig = plt.figure(figsize=(7.16, 6.2), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    # Panel (a): 2D Circular Diffraction (SAS)
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(i_circ / np.max(i_circ), extent=extent_circ, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_a.set_title("(a) Scalable Circular Diffraction ($z = 0.50$ m)", fontsize=8)
    ax_a.set_xlabel("$x$ [mm]")
    ax_a.set_ylabel("$y$ [mm]")

    # Panel (b): 2D Rectangular Diffraction (SAS)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(i_rect / np.max(i_rect), extent=extent_rect, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_b.set_title("(b) Scalable Rectangular Diffraction ($z = 0.50$ m)", fontsize=8)
    ax_b.set_xlabel("$x$ [mm]")
    ax_b.set_ylabel("$y$ [mm]")

    cbar_top = fig.colorbar(im_b, ax=[ax_a, ax_b], orientation="vertical", fraction=0.03, pad=0.04)
    cbar_top.set_label("Norm. Intensity $I/I_{\\max}$", fontsize=7)
    cbar_top.ax.tick_params(labelsize=6)

    # Panel (c): Airy Radial Profile Comparison
    ax_c = fig.add_subplot(gs[1, 0])
    r_mm = r_sas * 1e3
    r1_theory_mm = r1_theory * 1e3

    ax_c.plot(r_ana * 1e3, p_ana_n, color=COLORS["ref"], ls="--", lw=1.2, label="Analytical Airy")
    ax_c.plot(r_mm, p_sas_n, color=COLORS["sas"], lw=1.1, label="SAS Numerical")
    ax_c.plot(r_blas * 1e3, p_blas_n, color=COLORS["blas"], ls=":", lw=1.0, label="BL-ASM Numerical")
    ax_c.axvline(r1_theory_mm, color="#e41a1c", ls="-.", lw=0.9, label=f"1st Zero ({r1_theory_mm:.2f} mm)")

    ax_c.text(
        0.48, 0.65,
        f"1st zero err: {r1_err:.2f}%\nSAS profile err: {prof_l2_err:.2f}%",
        transform=ax_c.transAxes, fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.6)
    )

    ax_c.set_xlim(0, 3.5)
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.set_title("(c) Airy Radial Profile Across Scaled FOV", fontsize=8)
    ax_c.set_xlabel("Radius $r$ [mm]")
    ax_c.set_ylabel("Normalized Intensity")
    ax_c.legend(loc="upper right", frameon=False, fontsize=6.2)

    # Panel (d): Sinc Cross-Section Comparison
    ax_d = fig.add_subplot(gs[1, 1])
    coord_x_mm = grid_rect.x * 1e3
    coord_y_mm = grid_rect.y * 1e3

    ax_d.plot(coord_x_mm, x_ana, color=COLORS["ref"], ls="--", lw=1.2, label="Analytical $x$")
    ax_d.plot(coord_x_mm, x_num, color=COLORS["sas"], lw=1.1, label="SAS $x$")
    ax_d.plot(coord_y_mm, y_ana, color="#777777", ls=":", lw=1.1, label="Analytical $y$")
    ax_d.plot(coord_y_mm, y_num, color=COLORS["fresnel"], ls="-.", lw=1.0, label="SAS $y$")

    ax_d.text(
        0.52, 0.65,
        f"Slice $x$ err: {err_x:.2f}%\nSlice $y$ err: {err_y:.2f}%",
        transform=ax_d.transAxes, fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.6)
    )

    ax_d.set_xlim(-4.0, 4.0)
    ax_d.set_ylim(-0.02, 1.05)
    ax_d.set_title("(d) Rectangular Sinc Cross-Sections", fontsize=8)
    ax_d.set_xlabel("Coordinate [mm]")
    ax_d.set_ylabel("Normalized Intensity")
    ax_d.legend(loc="upper right", frameon=False, fontsize=6.2)

    fig.suptitle(
        "Analytical Fraunhofer Validation of Scalable Angular Spectrum ($z = 0.50$ m, $z \\gg a^2/\\lambda$)",
        fontsize=8.5, y=0.98
    )

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig04_method_selection_map(
    metrics_data: Optional[List[Dict[str, Any]]] = None,
    output_png: str = "assets/fig04_method_selection_map.png",
    output_pdf: str = "assets/fig04_method_selection_map.pdf",
    dpi: int = 400,
    quick: bool = False,
) -> None:
    """Generate Figure 4: 2x3 propagation method selection map across distance and sampling."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    from matplotlib.colors import LogNorm, ListedColormap
    from matplotlib.patches import Patch

    if metrics_data is None and os.path.exists("results/metrics.csv"):
        import csv
        with open("results/metrics.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("experiment") == "method_selection_map"]
            if rows:
                metrics_data = rows

    if metrics_data is None:
        from reproduce import run_method_selection_sweep
        metrics_data = run_method_selection_sweep(quick=quick)

    z_vals = sorted(list(set(float(r["z_m"]) for r in metrics_data)))
    dx_vals = sorted(list(set(float(r["dx_source_m"]) for r in metrics_data)))

    z_indices = {z: i for i, z in enumerate(z_vals)}
    dx_indices = {dx: i for i, dx in enumerate(dx_vals)}

    nz = len(z_vals)
    ndx = len(dx_vals)

    methods = ["fresnel", "asm", "blas", "sas"]
    err_maps = {m: np.full((ndx, nz), np.nan) for m in methods}
    methods = ["fresnel", "asm", "blas", "sas"]
    err_maps = {m: np.full((ndx, nz), np.nan) for m in methods}
    runtime_maps = {m: np.full((ndx, nz), np.nan) for m in methods}
    valid_maps = {m: np.full((ndx, nz), True) for m in methods}
    cov_maps = {m: np.full((ndx, nz), True) for m in methods}
    ref_conv_map = np.full((ndx, nz), True)
    ref_unc_map = np.full((ndx, nz), 0.005)
    mag_map = np.full((ndx, nz), 1.0)

    for r in metrics_data:
        m = r["method"]
        if m in methods:
            iz = z_indices[float(r["z_m"])]
            idx = dx_indices[float(r["dx_source_m"])]
            err = float(r["relative_l2_error"])
            err_maps[m][idx, iz] = err
            runtime_maps[m][idx, iz] = float(r["runtime_s"])
            val = str(r.get("valid", "True")).lower() == "true"
            valid_maps[m][idx, iz] = val
            cov = str(r.get("sufficient_coverage", "True")).lower() == "true"
            cov_maps[m][idx, iz] = cov
            ref_c = str(r.get("reference_converged", "True")).lower() == "true"
            ref_conv_map[idx, iz] = ref_c
            ref_unc_map[idx, iz] = float(r.get("ref_uncertainty", 0.005))
            if m == "sas":
                mag_val = float(r.get("magnification", float(r["dx_output_m"]) / float(r["dx_source_m"])))
                mag_map[idx, iz] = mag_val

    # Best map categories:
    # 0: Fresnel, 1: Std ASM, 2: BL-ASM, 3: SAS
    # 4: Tie / Indistinguishable
    # 5: Insufficient Coverage / Invalid
    # 6: Reference Unresolved
    best_map = np.full((ndx, nz), 6, dtype=int)
    for idx in range(ndx):
        for iz in range(nz):
            if not ref_conv_map[idx, iz]:
                best_map[idx, iz] = 6
                continue
            candidates = []
            for mi, m in enumerate(methods):
                if valid_maps[m][idx, iz] and cov_maps[m][idx, iz] and not np.isnan(err_maps[m][idx, iz]):
                    candidates.append((err_maps[m][idx, iz], mi))
            if candidates:
                candidates.sort()
                best_err, best_m = candidates[0]
                margin = max(0.005, ref_unc_map[idx, iz])
                tied = [m_idx for e, m_idx in candidates if abs(e - best_err) <= margin]
                if len(tied) > 1:
                    best_map[idx, iz] = 4
                else:
                    best_map[idx, iz] = best_m
            else:
                best_map[idx, iz] = 5

    fig = plt.figure(figsize=(7.16, 5.2), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.32)

    norm_err = LogNorm(vmin=0.005, vmax=1.0)
    cmap_err = "viridis"

    dx_labels_um = [f"{dx * 1e6:.1f}" for dx in dx_vals]
    z_labels = []
    for z in z_vals:
        if z < 0.01:
            z_labels.append(f"{z * 1e3:.0f}")
        elif z < 0.1:
            z_labels.append(f"{z * 1e3:.0f}")
        elif z < 1.0:
            z_labels.append(f"{z * 1e3:.0f}")
        else:
            z_labels.append("1000")

    titles = {
        "fresnel": "(a) Fresnel Relative $L_2$ Error",
        "asm": "(b) Standard ASM Relative $L_2$ Error",
        "blas": "(c) Band-Limited ASM Relative $L_2$ Error",
        "sas": "(d) Scalable AS (SAS) Relative $L_2$ Error",
    }

    panels = [("fresnel", 0, 0), ("asm", 0, 1), ("blas", 1, 0), ("sas", 1, 1)]

    for m, row, col in panels:
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            err_maps[m],
            origin="lower",
            norm=norm_err,
            cmap=cmap_err,
            aspect="auto",
        )
        ax.set_title(titles[m], fontsize=7.2)
        ax.set_yticks(range(ndx))
        if col == 0:
            ax.set_yticklabels(dx_labels_um, fontsize=6)
            ax.set_ylabel("$\\Delta x_s$ [$\\mu$m]", fontsize=6.8)
        else:
            ax.set_yticklabels([])

        step = max(1, nz // 5)
        ax.set_xticks(range(0, nz, step))
        ax.set_xticklabels([z_labels[i] for i in range(0, nz, step)], fontsize=6)
        if row == 1:
            ax.set_xlabel("Distance $z$ [mm]", fontsize=6.8)

        for idx in range(ndx):
            for iz in range(nz):
                if not ref_conv_map[idx, iz]:
                    ax.add_patch(Rectangle((iz - 0.5, idx - 0.5), 1, 1, fill=False, hatch="\\\\\\\\", edgecolor="#888888", lw=0.4))
                elif not valid_maps[m][idx, iz]:
                    ax.add_patch(Rectangle((iz - 0.5, idx - 0.5), 1, 1, fill=False, hatch="//", edgecolor="#fc8d62", lw=0.4))
                elif not cov_maps[m][idx, iz]:
                    ax.add_patch(Rectangle((iz - 0.5, idx - 0.5), 1, 1, fill=False, hatch="..", edgecolor="#7570b3", lw=0.4))

    cax = fig.add_axes([0.18, 0.94, 0.32, 0.018])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Relative $L_2$ Error Norm $\\epsilon_{L_2}$", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    ax_e = fig.add_subplot(gs[0, 2])
    disc_colors = ["#d95f02", "#386cb0", "#1b9e77", "#7570b3", "#4daf4a", "#fee08b", "#888888"]
    cmap_cat = ListedColormap(disc_colors)
    ax_e.imshow(best_map, origin="lower", cmap=cmap_cat, vmin=0, vmax=6, aspect="auto")
    ax_e.set_title("(e) Recommended Valid Method", fontsize=7.2)
    ax_e.set_yticks(range(ndx))
    ax_e.set_yticklabels([])
    step_e = max(1, nz // 5)
    ax_e.set_xticks(range(0, nz, step_e))
    ax_e.set_xticklabels([z_labels[i] for i in range(0, nz, step_e)], fontsize=6)

    for idx in range(ndx):
        for iz in range(nz):
            if best_map[idx, iz] == 6:
                ax_e.add_patch(Rectangle((iz - 0.5, idx - 0.5), 1, 1, fill=False, hatch="\\\\\\\\", edgecolor="#ffffff", lw=0.4))
            elif best_map[idx, iz] == 5:
                ax_e.add_patch(Rectangle((iz - 0.5, idx - 0.5), 1, 1, fill=False, hatch="..", edgecolor="#d95f02", lw=0.4))

    legend_elements = [
        Patch(facecolor="#d95f02", label="Fresnel"),
        Patch(facecolor="#386cb0", label="Std ASM"),
        Patch(facecolor="#1b9e77", label="BL-ASM"),
        Patch(facecolor="#7570b3", label="SAS"),
        Patch(facecolor="#4daf4a", label="Tie ($<0.5\\%$)"),
        Patch(facecolor="#fee08b", hatch="..", label="Low FOV"),
        Patch(facecolor="#888888", hatch="\\\\\\\\", label="Unresolved"),
    ]
    ax_e.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        fontsize=4.8,
        framealpha=0.85,
    )

    ax_f = fig.add_subplot(gs[1, 2])
    mean_runtimes = {m: np.nanmedian(runtime_maps[m], axis=0) * 1e3 for m in methods}
    ax_f.plot(z_vals, mean_runtimes["fresnel"], color=COLORS["fresnel"], lw=1.1, marker=".", ms=4, label="Fresnel")
    ax_f.plot(z_vals, mean_runtimes["asm"], color=COLORS["asm"], lw=1.1, marker="s", ms=3.5, label="Std ASM")
    ax_f.plot(z_vals, mean_runtimes["blas"], color=COLORS["blas"], lw=1.1, marker="^", ms=3.5, label="BL-ASM")
    ax_f.plot(z_vals, mean_runtimes["sas"], color=COLORS["sas"], lw=1.1, marker="d", ms=3.5, label="SAS")
    ax_f.set_xscale("log")
    ax_f.set_xlabel("Distance $z$ [m]", fontsize=6.8)
    ax_f.set_ylabel("Runtime [ms]", fontsize=6.8)
    ax_f.set_title("(f) Runtime & Magnification", fontsize=7.2)
    ax_f.tick_params(labelsize=6)
    ax_f.legend(loc="upper left", fontsize=5.2, framealpha=0.7)

    ax_f2 = ax_f.twinx()
    mid_dx_idx = min(1, ndx - 1)
    mag_curve = mag_map[mid_dx_idx, :]
    ax_f2.plot(z_vals, mag_curve, color="#e41a1c", ls="--", lw=1.0, label="Mag. $M$")
    ax_f2.axhline(1.0, color="#e41a1c", ls=":", lw=0.8)
    ax_f2.set_ylabel("SAS Mag. $M$", color="#e41a1c", fontsize=6.5)
    ax_f2.tick_params(axis="y", labelcolor="#e41a1c", labelsize=5.8)
    ax_f2.set_yscale("log")

    fig.suptitle(
        "FFT Propagation Method Selection Map: Physical-Coordinate Error and Validity",
        fontsize=8.5, y=0.98
    )

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig05_thin_lens_focus(
    lens_data: Optional[Dict[str, Any]] = None,
    output_png: str = "assets/fig05_thin_lens_focus.png",
    output_pdf: str = "assets/fig05_thin_lens_focus.pdf",
    dpi: int = 400,
    quick: bool = False,
) -> None:
    """Generate Figure 5: 2x3 Gaussian beam focusing through an ideal thin lens."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = (256, 256) if quick else (512, 512)
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4
    focal_length = 75e-3

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_0 = gaussian_field(grid, waist=w0)
    t_lens = thin_lens_phase(grid, wavelength, focal_length)
    u_lens = u_0 * t_lens

    _, _, z_focus_theory, w_focus_theory = gaussian_beam_abcd(w0, wavelength, focal_length, z=focal_length)

    if lens_data is not None:
        z_focus_meas = float(lens_data["focus_z_measured_m"])
        w_focus_meas = float(lens_data["waist_measured_m"])
        pos_err_pct = float(lens_data["focus_z_relative_error"]) * 100.0
        waist_err_pct = float(lens_data["waist_relative_error"]) * 100.0
    else:
        # Measure continuously via local quadratic fit around the discrete minimum
        z_dense = np.linspace(70.0e-3, 77.0e-3, 29 if quick else 71)
        w_dense = []
        for zi in z_dense:
            uzi = propagate_blas(u_lens, wavelength, z=zi, dx=dx, dy=dy, padding=2)
            wm = measure_beam_radius(np.abs(uzi)**2, grid.x, grid.y, roi_radius=4.0 * w_focus_theory)
            w_dense.append(wm)
        min_i = int(np.argmin(w_dense))
        fit_sl = slice(max(0, min_i - 2), min(len(z_dense), min_i + 3))
        p_fit = np.polyfit(z_dense[fit_sl], np.array(w_dense)[fit_sl], 2)
        z_raw = float(-p_fit[1] / (2.0 * p_fit[0]))
        w_raw = float(np.polyval(p_fit, z_raw))
        z_focus_meas = float(np.round(z_raw, 5))
        w_focus_meas = float(np.round(w_raw, 7))
        pos_err_pct = abs(z_focus_meas - z_focus_theory) / z_focus_theory * 100.0
        waist_err_pct = abs(w_focus_meas - w_focus_theory) / w_focus_theory * 100.0

    z_min, z_max = 10e-3, 140e-3
    nz_steps = 30 if quick else 90
    z_arr = np.linspace(z_min, z_max, nz_steps)

    x_line = grid.x
    cy = ny // 2
    i_xz = np.zeros((len(x_line), nz_steps), dtype=np.float64)
    w_meas_arr = np.zeros(nz_steps, dtype=np.float64)
    w_theory_arr = np.zeros(nz_steps, dtype=np.float64)

    for i, z_val in enumerate(z_arr):
        u_z = propagate_blas(u_lens, wavelength, z=z_val, dx=dx, dy=dy, padding=2)
        i_2d = np.abs(u_z)**2
        i_xz[:, i] = i_2d[cy, :]
        w_meas_arr[i] = measure_beam_radius(i_2d, grid.x, grid.y, roi_radius=4.0 * w_focus_theory)
        w_theory_arr[i], _, _, _ = gaussian_beam_abcd(w0, wavelength, focal_length, z=z_val)

    peak_focus_intensity = float(np.max(i_xz))
    i_xz_norm = i_xz / peak_focus_intensity

    z_before = 35e-3
    z_at = z_focus_meas
    z_after = 115e-3

    u_before = propagate_blas(u_lens, wavelength, z=z_before, dx=dx, dy=dy, padding=2)
    u_at = propagate_blas(u_lens, wavelength, z=z_at, dx=dx, dy=dy, padding=2)
    u_after = propagate_blas(u_lens, wavelength, z=z_after, dx=dx, dy=dy, padding=2)

    w_th_before, _, _, _ = gaussian_beam_abcd(w0, wavelength, focal_length, z_before)
    w_th_at, _, _, _ = gaussian_beam_abcd(w0, wavelength, focal_length, z_at)
    w_th_after, _, _, _ = gaussian_beam_abcd(w0, wavelength, focal_length, z_after)

    fig = plt.figure(figsize=(7.16, 4.8), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.36)

    extent_in_mm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(np.abs(u_0)**2, extent=extent_in_mm, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_a.set_title("(a) Input Gaussian ($w_0 = 0.30$ mm)", fontsize=7.5)
    ax_a.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_a.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_a.tick_params(labelsize=6)
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Intensity [a.u.]", fontsize=6)
    cbar_a.ax.tick_params(labelsize=5.5)

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(np.angle(t_lens), extent=extent_in_mm, origin="lower", cmap="coolwarm", vmin=-np.pi, vmax=np.pi)
    ax_b.set_title("(b) Thin Lens Phase ($f = 75$ mm)", fontsize=7.5)
    ax_b.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_b.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_b.tick_params(labelsize=6)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Phase [rad]", fontsize=6)
    cbar_b.ax.tick_params(labelsize=5.5)

    ax_c = fig.add_subplot(gs[0, 2])
    crop_x = np.abs(x_line) <= 0.8e-3
    x_sub_mm = x_line[crop_x] * 1e3
    i_sub = i_xz_norm[crop_x, :]

    extent_xz = [z_min * 1e3, z_max * 1e3, x_sub_mm[0], x_sub_mm[-1]]
    ax_c.imshow(i_sub, extent=extent_xz, origin="lower", cmap="inferno", aspect="auto", vmin=0, vmax=1.0)
    ax_c.plot(z_arr * 1e3, w_theory_arr * 1e3, color="#66c2a5", ls="--", lw=1.1, label="$\\pm w_{\\mathrm{ABCD}}(z)$")
    ax_c.plot(z_arr * 1e3, -w_theory_arr * 1e3, color="#66c2a5", ls="--", lw=1.1)
    ax_c.axvline(z_focus_theory * 1e3, color="white", ls=":", lw=0.9, label="Theory $z_f=73.54$ mm")
    ax_c.axvline(z_focus_meas * 1e3, color="#fc8d62", ls="-.", lw=0.9, label=f"Meas. $z_f={z_focus_meas*1e3:.2f}$ mm")
    ax_c.set_title("(c) Longitudinal Profile & Envelope", fontsize=7.5)
    ax_c.set_xlabel("$z$ [mm]", fontsize=6.8)
    ax_c.set_ylabel("$x$ [mm]", fontsize=6.8)
    ax_c.tick_params(labelsize=6)
    ax_c.legend(loc="upper right", fontsize=4.8, framealpha=0.85, facecolor="#222222", edgecolor="#555555", labelcolor="white")

    slices = [
        (gs[1, 0], "(d) Before Focus ($z = 35$ mm)", u_before, w_th_before, z_before),
        (gs[1, 1], f"(e) Focal Plane ($z = {z_at*1e3:.2f}$ mm)", u_at, w_th_at, z_at),
        (gs[1, 2], "(f) After Focus ($z = 115$ mm)", u_after, w_th_after, z_after),
    ]

    for ax_spec, title, u_slice, w_th, z_curr in slices:
        ax_s = fig.add_subplot(ax_spec)
        i_slice = np.abs(u_slice[cy, :])**2 / peak_focus_intensity
        i_th_exact = (w0 / w_th)**2 * np.exp(-2.0 * (x_line / w_th)**2)
        i_th_norm = i_th_exact / (w0 / w_focus_theory)**2

        x_um = x_line * 1e6
        ax_s.plot(x_um, i_slice, color="#1b9e77", lw=1.2, label="BL-ASM")
        ax_s.plot(x_um, i_th_norm, color="#222222", ls="--", lw=1.0, label="ABCD")
        ax_s.set_xlim(-300, 300)
        ax_s.set_title(title, fontsize=7.2)
        ax_s.set_xlabel("$x$ [$\\mu$m]", fontsize=6.8)
        ax_s.set_ylabel("Norm. Intensity", fontsize=6.8)
        ax_s.tick_params(labelsize=6)
        if "Focal Plane" in title:
            ax_s.text(
                0.03, 0.60,
                f"Theory $z_f = 73.54$ mm\n"
                f"Meas. $z_f = {z_focus_meas*1e3:.2f} \\pm 0.01$ mm\n"
                f"Theory $w_0' = 41.9\\,\\mu$m\n"
                f"Meas. $w_0' = {w_focus_meas*1e6:.1f} \\pm 0.1\\,\\mu$m\n"
                f"Waist diff: $\\approx {waist_err_pct:.2f}\\%$\n"
                f"Pos. diff: $\\approx {pos_err_pct:.3f}\\%$",
                transform=ax_s.transAxes, fontsize=4.8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f7f7f7", edgecolor="#cccccc", lw=0.6)
            )
        ax_s.legend(loc="upper right", fontsize=5.0, framealpha=0.7)

    fig.suptitle(
        "Thin-Lens Focusing of a Gaussian Beam: Independent ABCD q-Parameter Validation",
        fontsize=8.5, y=0.98
    )

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig06_vortex_beam(
    vortex_data: Optional[Dict[str, Any]] = None,
    output_png: str = "assets/fig06_vortex_beam.png",
    output_pdf: str = "assets/fig06_vortex_beam.pdf",
    dpi: int = 400,
    quick: bool = False,
) -> None:
    """Generate Figure 6: 2x3 optical vortex beam and topological charge conservation."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    from scipy.interpolate import RegularGridInterpolator

    nx, ny = (256, 256) if quick else (512, 512)
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    w0 = 3.0e-4
    z_prop = 0.25

    grid = create_grid(nx=nx, ny=ny, dx=dx, dy=dy)
    u_gauss = gaussian_field(grid, waist=w0)

    ell_main = 1
    t_vortex_1 = spiral_phase(grid, charge=ell_main)
    u_in_1 = u_gauss * t_vortex_1
    u_out_1 = propagate_blas(u_in_1, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)

    r_contour = 1.5e-4
    n_contour = 360
    theta_contour = np.linspace(0, 2.0 * np.pi, n_contour, endpoint=False)
    x_c = r_contour * np.cos(theta_contour)
    y_c = r_contour * np.sin(theta_contour)

    interp_r = RegularGridInterpolator((grid.y, grid.x), np.real(u_out_1), bounds_error=False, fill_value=0.0)
    interp_i = RegularGridInterpolator((grid.y, grid.x), np.imag(u_out_1), bounds_error=False, fill_value=0.0)
    pts = np.column_stack([y_c, x_c])
    u_c = interp_r(pts) + 1j * interp_i(pts)
    phase_wrapped = np.angle(u_c)
    phase_unwrapped = np.unwrap(phase_wrapped)

    ell_vals = [1, 2, 3]
    if vortex_data is not None and "charges" in vortex_data:
        charge_entries = {c["charge_requested"]: c for c in vortex_data["charges"]}
        q_in_meas = [charge_entries.get(ell, {}).get("charge_input_measured", float(ell)) for ell in ell_vals]
        q_out_meas = [charge_entries.get(ell, {}).get("charge_output_measured", float(ell)) for ell in ell_vals]
    else:
        q_in_meas = []
        q_out_meas = []
        for ell in ell_vals:
            t_v = spiral_phase(grid, charge=ell)
            u_v = u_gauss * t_v
            q_in = topological_charge(u_v, grid, radius=r_contour, num_points=360)
            u_p = propagate_blas(u_v, wavelength, z=z_prop, dx=dx, dy=dy, padding=2)
            q_out = topological_charge(u_p, grid, radius=r_contour, num_points=360)
            q_in_meas.append(q_in)
            q_out_meas.append(q_out)

    fig = plt.figure(figsize=(7.16, 4.8), dpi=dpi)
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.36)

    extent_mm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(np.abs(u_in_1), extent=extent_mm, origin="lower", cmap="magma", vmin=0, vmax=1)
    circle = plt.Circle((0, 0), r_contour * 1e3, fill=False, color="#ffffff", ls="--", lw=1.1, label=f"Contour $R={r_contour*1e3:.2f}$ mm")
    ax_a.add_patch(circle)
    ax_a.set_title("(a) Input Amplitude $|U_0|$ ($\\ell=1$)", fontsize=7.5)
    ax_a.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_a.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_a.tick_params(labelsize=6)
    ax_a.legend(loc="upper right", fontsize=5.0, framealpha=0.85, facecolor="#222222", edgecolor="#555555", labelcolor="white")
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cbar_a.set_label("Amplitude", fontsize=6)
    cbar_a.ax.tick_params(labelsize=5.5)

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(np.angle(u_in_1), extent=extent_mm, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_b.set_title("(b) Input Phase $\\arg(U_0)$", fontsize=7.5)
    ax_b.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_b.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_b.tick_params(labelsize=6)
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Phase [rad]", fontsize=6)
    cbar_b.ax.tick_params(labelsize=5.5)

    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(np.abs(u_out_1)**2 / np.max(np.abs(u_out_1)**2), extent=extent_mm, origin="lower", cmap="inferno", vmin=0, vmax=1)
    ax_c.set_title(f"(c) Propagated Donut ($z = {z_prop:.2f}$ m)", fontsize=7.5)
    ax_c.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_c.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_c.tick_params(labelsize=6)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Norm. Intensity", fontsize=6)
    cbar_c.ax.tick_params(labelsize=5.5)

    ax_d = fig.add_subplot(gs[1, 0])
    im_d = ax_d.imshow(np.angle(u_out_1), extent=extent_mm, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_d.set_title("(d) Propagated Phase $\\arg(U)$", fontsize=7.5)
    ax_d.set_xlabel("$x$ [mm]", fontsize=6.8)
    ax_d.set_ylabel("$y$ [mm]", fontsize=6.8)
    ax_d.tick_params(labelsize=6)
    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("Phase [rad]", fontsize=6)
    cbar_d.ax.tick_params(labelsize=5.5)

    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(theta_contour, phase_wrapped, color="#386cb0", ls=":", lw=1.0, label="Wrapped $\\psi(\\theta)$")
    ax_e.plot(theta_contour, phase_unwrapped - phase_unwrapped[0], color="#1b9e77", lw=1.3, label="Unwrapped $\\Delta\\psi$")
    ax_e.axhline(2.0 * np.pi, color="#e41a1c", ls="--", lw=0.9, label="$2\\pi$ Target")
    ax_e.set_title("(e) Phase Winding Around Contour", fontsize=7.5)
    ax_e.set_xlabel("Azimuthal Angle $\\theta$ [rad]", fontsize=6.8)
    ax_e.set_ylabel("Accumulated Phase [rad]", fontsize=6.8)
    ax_e.set_xticks([0, np.pi, 2.0 * np.pi])
    ax_e.set_xticklabels(["$0$", "$\\pi$", "$2\\pi$"], fontsize=6)
    ax_e.set_yticks([0, np.pi, 2.0 * np.pi])
    ax_e.set_yticklabels(["$0$", "$\\pi$", "$2\\pi$"], fontsize=6)
    ax_e.set_ylim(-0.5, 7.0)
    ax_e.tick_params(labelsize=6)
    ax_e.legend(loc="upper left", fontsize=5.2, framealpha=0.8)

    ax_f = fig.add_subplot(gs[1, 2])
    x_indices = np.arange(len(ell_vals))
    bar_width = 0.32
    rects1 = ax_f.bar(x_indices - bar_width / 2, q_in_meas, bar_width, label="Input $Q_{\\mathrm{in}}$", color="#386cb0", alpha=0.85)
    rects2 = ax_f.bar(x_indices + bar_width / 2, q_out_meas, bar_width, label="Prop. $Q_{\\mathrm{out}}$", color="#1b9e77", alpha=0.85)

    ax_f.set_xticks(x_indices)
    ax_f.set_xticklabels([f"$\\ell = {ell}$" for ell in ell_vals], fontsize=6.5)
    ax_f.set_ylabel("Measured Charge $Q$", fontsize=6.8)
    ax_f.set_title("(f) Topological Charge Conservation", fontsize=7.5)
    ax_f.set_ylim(0, 3.8)
    ax_f.tick_params(labelsize=6)
    ax_f.legend(loc="upper left", fontsize=5.5, framealpha=0.7)

    for rect in rects2:
        h = rect.get_height()
        ax_f.text(rect.get_x() + rect.get_width() / 2.0, h + 0.08, f"{h:.3f}", ha="center", va="bottom", fontsize=5.5)

    fig.suptitle(
        "Phase-Only Optical Vortex: Free-Space Propagation and Topological Charge Conservation",
        fontsize=8.5, y=0.98
    )

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_primary_loop_gif(
    output_gif: str = "assets/diffraction_loop.gif",
    num_steps: int = 36,
    fps: int = 14,
) -> int:
    """Generate primary seamless 2x2 loop GIF: Static apertures above evolving diffraction fields."""
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    nx, ny = 256, 256
    dx, dy = 6e-6, 6e-6
    wavelength = 532e-9
    grid = create_grid(nx, ny, dx, dy)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    radius = 1.5e-4
    w_rect, h_rect = 3.0e-4, 4.5e-4
    u_circ_in = circular_aperture(grid, radius=radius)
    u_rect_in = rectangular_aperture(grid, width=w_rect, height=h_rect)

    z_min = 0.005
    z_max = 0.250
    z_forward = np.geomspace(z_min, z_max, num_steps)
    z_cycle = np.concatenate([z_forward, z_forward[-2:0:-1]])

    frames = []

    for z_val in z_cycle:
        u_circ = propagate_blas(u_circ_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)
        u_rect = propagate_blas(u_rect_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)

        i_circ_disp = np.sqrt(np.abs(u_circ)**2 / np.max(np.abs(u_circ)**2))
        i_rect_disp = np.sqrt(np.abs(u_rect)**2 / np.max(np.abs(u_rect)**2))

        fig = plt.figure(figsize=(6.2, 5.2), dpi=120)
        fig.patch.set_facecolor("#111111")
        gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

        # Top row: Static apertures
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.imshow(np.abs(u_circ_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax_a.set_title("Circular Aperture ($2a = 0.30$ mm)", fontsize=7.5, color="white")
        ax_a.set_ylabel("$y$ [mm]", color="white")
        ax_a.tick_params(colors="white")

        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.imshow(np.abs(u_rect_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax_b.set_title("Rectangular ($0.30 \\times 0.45$ mm)", fontsize=7.5, color="white")
        ax_b.tick_params(colors="white")

        # Bottom row: Evolving diffraction
        ax_c = fig.add_subplot(gs[1, 0])
        ax_c.imshow(i_circ_disp, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_c.set_title("Circular Diffraction", fontsize=7.5, color="white")
        ax_c.set_xlabel("$x$ [mm]", color="white")
        ax_c.set_ylabel("$y$ [mm]", color="white")
        ax_c.tick_params(colors="white")

        ax_d = fig.add_subplot(gs[1, 1])
        ax_d.imshow(i_rect_disp, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_d.set_title("Rectangular Diffraction", fontsize=7.5, color="white")
        ax_d.set_xlabel("$x$ [mm]", color="white")
        ax_d.tick_params(colors="white")

        fresnel_num = (radius**2) / (wavelength * z_val)
        fig.suptitle(
            f"Free-Space Propagation: $z = {z_val * 1e3:5.1f}$ mm   |   $N_F = {fresnel_num:4.2f}$",
            fontsize=8.5, color="#66c2a5", y=0.98
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
        frames.append(img)
        plt.close(fig)

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return len(frames)


def generate_error_loop_gif(
    output_gif: str = "assets/error_loop.gif",
    num_steps: int = 28,
    fps: int = 12,
) -> int:
    """Generate secondary 2x2 loop GIF: Standard ASM vs SAS intensity and error evolution."""
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    nx, ny = 256, 256
    dx, dy = 6e-6, 6e-6
    wavelength = 532e-9
    radius = 1.2e-4

    grid = create_grid(nx, ny, dx, dy)
    extent_asm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]
    u_in = circular_aperture(grid, radius=radius)

    z_min = 0.35
    z_max = 0.85
    z_forward = np.linspace(z_min, z_max, num_steps)
    z_cycle = np.concatenate([z_forward, z_forward[-2:0:-1]])

    frames = []

    for z_val in z_cycle:
        u_asm = propagate_asm(u_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)
        u_sas, grid_sas = propagate_sas(u_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)
        i_ref_asm = airy_pattern(grid, wavelength, z=z_val, radius=radius)
        i_ref_sas = airy_pattern(grid_sas, wavelength, z=z_val, radius=radius)

        extent_sas = [grid_sas.x[0] * 1e3, grid_sas.x[-1] * 1e3, grid_sas.y[0] * 1e3, grid_sas.y[-1] * 1e3]

        i_asm = np.abs(u_asm)**2
        i_sas = np.abs(u_sas)**2

        ref_max_asm = np.max(i_ref_asm)
        ref_max_sas = np.max(i_ref_sas)
        i_asm_n = i_asm / ref_max_asm
        i_sas_n = i_sas / ref_max_sas
        i_ref_asm_n = i_ref_asm / ref_max_asm
        i_ref_sas_n = i_ref_sas / ref_max_sas

        err_asm = np.abs(i_asm_n - i_ref_asm_n)
        err_sas = np.abs(i_sas_n - i_ref_sas_n)

        # Fair shared region calculation
        x_max = grid.Sx / 2.0
        l2_asm = relative_l2_error(i_asm, i_ref_asm) * 100.0
        l2_sas_shared = compute_shared_region_l2_error(i_sas, grid_sas, i_ref_sas, x_max=x_max, y_max=x_max) * 100.0

        fig = plt.figure(figsize=(6.2, 5.2), dpi=120)
        fig.patch.set_facecolor("#111111")
        gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

        # Top Row: Intensities
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.imshow(i_asm_n, extent=extent_asm, origin="lower", cmap="inferno", vmin=0, vmax=1.0)
        ax_a.set_title("Standard ASM (Fixed Grid)", fontsize=7.5, color="white")
        ax_a.set_ylabel("$y$ [mm]", color="white")
        ax_a.tick_params(colors="white")

        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.imshow(i_sas_n, extent=extent_sas, origin="lower", cmap="inferno", vmin=0, vmax=1.0)
        ax_b.set_title("Scalable Angular Spectrum (SAS)", fontsize=7.5, color="white")
        ax_b.tick_params(colors="white")

        # Bottom Row: Error Maps
        ax_c = fig.add_subplot(gs[1, 0])
        ax_c.imshow(err_asm, extent=extent_asm, origin="lower", cmap="magma", vmin=0, vmax=0.18)
        ax_c.set_title(f"ASM Error (L2: {l2_asm:.1f}%)", fontsize=7.5, color="#fc8d62")
        ax_c.set_xlabel("$x$ [mm]", color="white")
        ax_c.set_ylabel("$y$ [mm]", color="white")
        ax_c.tick_params(colors="white")

        ax_d = fig.add_subplot(gs[1, 1])
        ax_d.imshow(err_sas, extent=extent_sas, origin="lower", cmap="magma", vmin=0, vmax=0.18)
        ax_d.set_title(f"SAS Shared Error (L2: {l2_sas_shared:.1f}%)", fontsize=7.5, color="#66c2a5")
        ax_d.set_xlabel("$x$ [mm]", color="white")
        ax_d.tick_params(colors="white")

        fov_sas_mm = grid_sas.Sx * 1e3
        fig.suptitle(
            f"Distance: $z = {z_val * 1e3:5.0f}$ mm   |   SAS FOV: {fov_sas_mm:.1f} mm",
            fontsize=8.5, color="white", y=0.98
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba).convert("P", palette=Image.ADAPTIVE)
        frames.append(img)
        plt.close(fig)

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return len(frames)


if __name__ == "__main__":
    print("Generating Figure 1 (Apertures above diffraction)...")
    generate_fig01_aperture_diffraction()
    print("Generating Figure 2 (Methods and error maps)...")
    generate_fig02_methods_errors()
    print("Generating Figure 3 (Results and profiles)...")
    generate_fig03_results_profiles()
    print("Generating Figure 4 (Method selection map)...")
    generate_fig04_method_selection_map()
    print("Generating Figure 5 (Thin-lens focusing)...")
    generate_fig05_thin_lens_focus()
    print("Generating Figure 6 (Optical vortex beam)...")
    generate_fig06_vortex_beam()
    print("Generating Primary diffraction loop GIF...")
    n_frames1 = generate_primary_loop_gif()
    print(f"Primary GIF complete ({n_frames1} frames).")
    print("Generating Error evolution loop GIF...")
    n_frames2 = generate_error_loop_gif()
    print(f"Error GIF complete ({n_frames2} frames).")
