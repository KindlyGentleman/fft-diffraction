"""Scientific visualization module for FFT scalar diffraction experiments.

Generates:
1. Fig 1: 2x2 Input apertures aligned above corresponding diffraction fields.
2. Fig 2: 2x4 Fair comparison of Fresnel, ASM, BL-ASM, and SAS with error maps.
3. Fig 3: 2x2 Analytical Fraunhofer validation with quantitative cross-sections.
4. Fig 4: 4x2 Gallery A: Circular, Rectangular, Double slit, and Annular apertures.
5. Fig 5: 4x2 Gallery B: Gaussian field, Binary grating, Triangular, and Off-axis circular.
6. Fig 6: 2x5 Propagation distance sweep and grid scaling comparison.
7. Primary GIF: Animated free-space propagation loop.
8. Error GIF: Animated ASM vs SAS error evolution loop.
"""

import os
from typing import Tuple, List, Optional
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
    sas_output_scaling,
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
    ax_e.text(0.04, 0.88, f"Shared $L_2$: {l2_fres_shared:.1f}%\nFOV: 2.56 mm (clipped)",
              transform=ax_e.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    ax_f = fig.add_subplot(gs[1, 1])
    ax_f.imshow(err_asm, extent=extent_fixed, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_f.set_title("(f) ASM Error (Aliasing)", fontsize=7.5)
    ax_f.set_xlabel("$x$ [mm]")
    ax_f.tick_params(labelsize=6)
    ax_f.text(0.04, 0.88, f"Shared $L_2$: {l2_asm_shared:.1f}%\nFOV: 2.56 mm (clipped)",
              transform=ax_f.transAxes, fontsize=5.4, color="white",
              bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    ax_g = fig.add_subplot(gs[1, 2])
    ax_g.imshow(err_blas, extent=extent_fixed, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_g.set_title("(g) BL-ASM Error", fontsize=7.5)
    ax_g.set_xlabel("$x$ [mm]")
    ax_g.tick_params(labelsize=6)
    ax_g.text(0.04, 0.88, f"Shared $L_2$: {l2_blas_shared:.1f}%\nFOV: 2.56 mm (clipped)",
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
    ax_h.text(0.04, 0.88, f"Shared: {l2_sas_shared:.1f}% | Full: {l2_sas_full:.1f}%\nFOV: 42.56 mm (full)",
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


def generate_fig04_aperture_gallery_a(
    output_png: str = "assets/fig04_aperture_gallery_a.png",
    output_pdf: str = "assets/fig04_aperture_gallery_a.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 4: Diffraction Gallery A (Circular, Rectangular, Double Slit, Annular)."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    grid = create_grid(nx, ny, dx, dy)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    # 1. Circular aperture (Airy diffraction)
    u1_in = circular_aperture(grid, radius=1.5e-4)
    z1 = 0.40
    u1_out = propagate_blas(u1_in, wavelength, z=z1, dx=dx, dy=dy, padding=2)
    i1 = np.abs(u1_out)**2

    # 2. Rectangular aperture (sinc-squared diffraction)
    u2_in = rectangular_aperture(grid, width=3.0e-4, height=4.5e-4)
    z2 = 0.40
    u2_out = propagate_blas(u2_in, wavelength, z=z2, dx=dx, dy=dy, padding=2)
    i2 = np.abs(u2_out)**2

    # 3. Double slit (interference fringes modulated by envelope)
    u3_in = double_slit_aperture(grid, slit_width=5e-5, slit_height=4e-4, separation=2e-4)
    z3 = 0.20
    u3_out = propagate_blas(u3_in, wavelength, z=z3, dx=dx, dy=dy, padding=2)
    i3 = np.abs(u3_out)**2

    # 4. Annular aperture (ring-shaped pupil with central obscuration)
    u4_in = annular_aperture(grid, r_inner=8e-5, r_outer=1.8e-4)
    z4 = 0.40
    u4_out = propagate_blas(u4_in, wavelength, z=z4, dx=dx, dy=dy, padding=2)
    i4 = np.abs(u4_out)**2

    fig = plt.figure(figsize=(6.8, 9.2), dpi=dpi)
    gs = GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.25)

    cases = [
        (0, u1_in, i1, "(a) Circular Aperture ($2a = 0.30$ mm)", f"(b) Airy Diffraction ($z = {z1:.2f}$ m)",
         "Classic Airy pattern: central disc with dark ring at $r_1 = 0.87$ mm"),
        (1, u2_in, i2, "(c) Rectangular Aperture ($0.30 \\times 0.45$ mm)", f"(d) Sinc-Squared Diffraction ($z = {z2:.2f}$ m)",
         "2D separable sinc-squared pattern with orthogonal sidelobes"),
        (2, u3_in, i3, "(e) Double Slit ($w = 0.05$ mm, $d = 0.20$ mm)", f"(f) Young Interference ($z = {z3:.2f}$ m)",
         "High-contrast fringes modulated by single-slit diffraction envelope"),
        (3, u4_in, i4, "(g) Annular Aperture ($r_{\\mathrm{in}}=0.08, r_{\\mathrm{out}}=0.18$ mm)", f"(h) Obscured Ring Diffraction ($z = {z4:.2f}$ m)",
         "Central obscuration sharpens core while boosting outer ring intensity"),
    ]

    last_im = None
    for row, u_in, i_out, t_in, t_out, desc in cases:
        ax_in = fig.add_subplot(gs[row, 0])
        ax_in.imshow(np.abs(u_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax_in.set_title(t_in, fontsize=7.5)
        ax_in.set_ylabel("$y$ [mm]")
        if row == 3:
            ax_in.set_xlabel("$x$ [mm]")
        ax_in.tick_params(labelsize=6)

        ax_out = fig.add_subplot(gs[row, 1])
        i_norm = i_out / np.max(i_out)
        last_im = ax_out.imshow(i_norm, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_out.set_title(t_out, fontsize=7.5)
        if row == 3:
            ax_out.set_xlabel("$x$ [mm]")
        ax_out.tick_params(labelsize=6)
        ax_out.text(0.03, 0.05, desc, transform=ax_out.transAxes, fontsize=5.6, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    cbar = fig.colorbar(last_im, ax=fig.axes, orientation="horizontal", fraction=0.025, pad=0.08, aspect=40)
    cbar.set_label("Normalized Intensity $I / I_{\\max}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Diffraction Example Gallery A: Canonical Apertures and Wave Fields", fontsize=8.8, y=0.99)

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig05_aperture_gallery_b(
    output_png: str = "assets/fig05_aperture_gallery_b.png",
    output_pdf: str = "assets/fig05_aperture_gallery_b.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 5: Diffraction Gallery B (Gaussian, Binary Grating, Triangular, Off-Axis)."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    grid = create_grid(nx, ny, dx, dy)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    # 5. Gaussian beam (smooth beam propagation)
    u5_in = gaussian_field(grid, waist=1.2e-4)
    z5 = 0.20
    u5_out = propagate_blas(u5_in, wavelength, z=z5, dx=dx, dy=dy, padding=2)
    i5 = np.abs(u5_out)**2

    # 6. Binary transmission grating (multiple diffraction orders)
    u6_in = binary_grating(grid, period=6e-5, duty_cycle=0.5, width=5e-4, height=4e-4)
    z6 = 0.12
    u6_out = propagate_blas(u6_in, wavelength, z=z6, dx=dx, dy=dy, padding=2)
    i6 = np.abs(u6_out)**2

    # 7. Triangular aperture (non-centrosymmetric geometry, 6-pointed star)
    u7_in = triangular_aperture(grid, side_length=3.5e-4)
    z7 = 0.25
    u7_out = propagate_blas(u7_in, wavelength, z=z7, dx=dx, dy=dy, padding=2)
    i7 = np.abs(u7_out)**2

    # 8. Off-axis circular aperture (shifted-field behavior)
    u8_in = shifted_circular_aperture(grid, radius=1.2e-4, x_offset=2.5e-4, y_offset=2.5e-4)
    z8 = 0.25
    u8_out = propagate_blas(u8_in, wavelength, z=z8, dx=dx, dy=dy, padding=2)
    i8 = np.abs(u8_out)**2

    fig = plt.figure(figsize=(6.8, 9.2), dpi=dpi)
    gs = GridSpec(4, 2, figure=fig, hspace=0.38, wspace=0.25)

    cases = [
        (0, u5_in, i5, "(a) Gaussian Field ($w_0 = 0.12$ mm)", f"(b) Smooth Beam Expansion ($z = {z5:.2f}$ m)",
         "Smooth monotonic expansion without sharp edge diffraction ripples"),
        (1, u6_in, i6, "(c) Binary Grating ($\\Lambda = 0.06$ mm)", f"(d) Diffraction Orders ($z = {z6:.2f}$ m)",
         "Discrete diffraction orders ($0, \\pm 1$) separated by $\\Delta \\theta \\approx \\lambda / \\Lambda$"),
        (2, u7_in, i7, "(e) Triangular Aperture (Side $0.35$ mm)", f"(f) Six-Pointed Diffraction Star ($z = {z7:.2f}$ m)",
         "Six-fold diffraction spikes reflecting 3-fold polygonal edge normals"),
        (3, u8_in, i8, "(g) Off-Axis Aperture ($x_0 = y_0 = 0.25$ mm)", f"(h) Shifted Diffraction Pattern ($z = {z8:.2f}$ m)",
         "Lateral shift in input plane preserves envelope under Fraunhofer propagation"),
    ]

    last_im = None
    for row, u_in, i_out, t_in, t_out, desc in cases:
        ax_in = fig.add_subplot(gs[row, 0])
        ax_in.imshow(np.abs(u_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax_in.set_title(t_in, fontsize=7.5)
        ax_in.set_ylabel("$y$ [mm]")
        if row == 3:
            ax_in.set_xlabel("$x$ [mm]")
        ax_in.tick_params(labelsize=6)

        ax_out = fig.add_subplot(gs[row, 1])
        i_norm = i_out / np.max(i_out)
        last_im = ax_out.imshow(i_norm, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax_out.set_title(t_out, fontsize=7.5)
        if row == 3:
            ax_out.set_xlabel("$x$ [mm]")
        ax_out.tick_params(labelsize=6)
        ax_out.text(0.03, 0.05, desc, transform=ax_out.transAxes, fontsize=5.6, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, lw=0))

    cbar = fig.colorbar(last_im, ax=fig.axes, orientation="horizontal", fraction=0.025, pad=0.08, aspect=40)
    cbar.set_label("Normalized Intensity $I / I_{\\max}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Diffraction Example Gallery B: Structured Waves and Polygonal Shapes", fontsize=8.8, y=0.99)

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig06_distance_sweep(
    output_png: str = "assets/fig06_distance_sweep.png",
    output_pdf: str = "assets/fig06_distance_sweep.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 6: Propagation distance and scaling comparison across near and far fields."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4

    grid_in = create_grid(nx, ny, dx, dy)
    u_in = circular_aperture(grid_in, radius=radius)
    extent_fixed = [grid_in.x[0] * 1e3, grid_in.x[-1] * 1e3, grid_in.y[0] * 1e3, grid_in.y[-1] * 1e3]

    distances = [0.005, 0.020, 0.050, 0.200, 0.800]

    fig = plt.figure(figsize=(7.16, 3.8), dpi=dpi)
    gs = GridSpec(2, 5, figure=fig, hspace=0.36, wspace=0.30)

    for col, z in enumerate(distances):
        u_blas = propagate_blas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)
        u_sas, grid_sas = propagate_sas(u_in, wavelength, z=z, dx=dx, dy=dy, padding=2)

        i_blas = np.abs(u_blas)**2
        i_sas = np.abs(u_sas)**2

        i_blas_norm = i_blas / np.max(i_blas)
        i_sas_norm = i_sas / np.max(i_sas)

        extent_sas = [grid_sas.x[0] * 1e3, grid_sas.x[-1] * 1e3, grid_sas.y[0] * 1e3, grid_sas.y[-1] * 1e3]
        fresnel_num = (radius**2) / (wavelength * z)
        mag_m = grid_sas.dx / dx

        # Row 1: Fixed-Grid BL-ASM
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(i_blas_norm, extent=extent_fixed, origin="lower", cmap="inferno", vmin=0, vmax=1)
        z_str = f"{z * 1e3:.0f} mm" if z < 0.10 else f"{z * 1e3:.0f} mm"
        ax_top.set_title(f"$z = {z_str}$\n$N_F = {fresnel_num:.2f}$", fontsize=7)
        if col == 0:
            ax_top.set_ylabel("BL-ASM (Fixed)\n$y$ [mm]", fontsize=6.8)
        ax_top.tick_params(labelsize=5.5)

        # Row 2: Scalable Angular Spectrum
        ax_bot = fig.add_subplot(gs[1, col])
        im_bot = ax_bot.imshow(i_sas_norm, extent=extent_sas, origin="lower", cmap="inferno", vmin=0, vmax=1)
        fov_str = f"{grid_sas.Sx * 1e3:.2f} mm" if grid_sas.Sx < 0.01 else f"{grid_sas.Sx * 1e3:.1f} mm"
        ax_bot.set_title(f"FOV: {fov_str} ($m={mag_m:.2f}$)", fontsize=6.5)
        ax_bot.set_xlabel("$x$ [mm]", fontsize=6.8)
        if col == 0:
            ax_bot.set_ylabel("SAS (Scaled)\n$y$ [mm]", fontsize=6.8)
        ax_bot.tick_params(labelsize=5.5)

    cbar = fig.colorbar(im_bot, ax=fig.axes, orientation="horizontal", fraction=0.035, pad=0.18, aspect=45)
    cbar.set_label("Normalized Intensity $I / I_{\\max}$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Evolution of Circular Aperture Diffraction: Fixed-Grid BL-ASM vs Scalable Angular Spectrum",
        fontsize=8.5, y=0.99
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

    u_circ_in = circular_aperture(grid, radius=1.2e-4)
    u_rect_in = rectangular_aperture(grid, width=2.2e-4, height=3.2e-4)

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
        ax_a.set_title("Circular Aperture ($2a = 0.24$ mm)", fontsize=7.5, color="white")
        ax_a.set_ylabel("$y$ [mm]", color="white")
        ax_a.tick_params(colors="white")

        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.imshow(np.abs(u_rect_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax_b.set_title("Rectangular ($0.22 \\times 0.32$ mm)", fontsize=7.5, color="white")
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

        fresnel_num = (1.2e-4)**2 / (wavelength * z_val)
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

        fov_sas_mm = (grid_sas.x[-1] - grid_sas.x[0]) * 1e3
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
    print("Generating Figure 4 (Gallery A)...")
    generate_fig04_aperture_gallery_a()
    print("Generating Figure 5 (Gallery B)...")
    generate_fig05_aperture_gallery_b()
    print("Generating Figure 6 (Distance sweep)...")
    generate_fig06_distance_sweep()
    print("Generating Primary diffraction loop GIF...")
    n_frames1 = generate_primary_loop_gif()
    print(f"Primary GIF complete ({n_frames1} frames).")
    print("Generating Error evolution loop GIF...")
    n_frames2 = generate_error_loop_gif()
    print(f"Error GIF complete ({n_frames2} frames).")
