"""Scientific visualization module for FFT scalar diffraction experiments.

Generates:
1. Fig 1: 2x2 Apertures aligned above their corresponding diffraction fields.
2. Fig 2: 2x2 Standard ASM vs Band-Limited ASM intensity and error maps.
3. Fig 3: 2x2 Propagated intensities with quantitative analytical profiles.
4. Primary GIF: 2x2 Static apertures above animated diffraction fields.
5. Error GIF: 2x2 Standard ASM vs BL-ASM intensity and error evolution.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from diffraction import (
    create_grid,
    circular_aperture,
    rectangular_aperture,
    propagate_asm,
    propagate_blas,
    propagate_sas,
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    radial_profile,
    find_first_radial_minimum,
    relative_l2_error,
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

    # Shared colorbar for bottom diffraction panels
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
    """Generate Figure 2: Standard ASM vs Scalable Angular Spectrum (SAS) with error maps."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4
    z_eval = 0.80

    grid = create_grid(nx, ny, dx, dy)
    extent_asm = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    u_in = circular_aperture(grid, radius=radius)
    u_asm = propagate_asm(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)
    u_sas, grid_sas = propagate_sas(u_in, wavelength, z=z_eval, dx=dx, dy=dy, padding=2)

    extent_sas = [grid_sas.x[0] * 1e3, grid_sas.x[-1] * 1e3, grid_sas.y[0] * 1e3, grid_sas.y[-1] * 1e3]

    i_ref_asm = airy_pattern(grid, wavelength, z=z_eval, radius=radius)
    i_ref_sas = airy_pattern(grid_sas, wavelength, z=z_eval, radius=radius)

    i_asm = np.abs(u_asm)**2
    i_sas = np.abs(u_sas)**2

    i_asm_disp = i_asm / np.max(i_ref_asm)
    i_ref_asm_disp = i_ref_asm / np.max(i_ref_asm)
    err_asm = np.abs(i_asm_disp - i_ref_asm_disp)

    i_sas_disp = i_sas / np.max(i_ref_sas)
    i_ref_sas_disp = i_ref_sas / np.max(i_ref_sas)
    err_sas = np.abs(i_sas_disp - i_ref_sas_disp)

    l2_asm = relative_l2_error(i_asm, i_ref_asm) * 100.0
    l2_sas = relative_l2_error(i_sas, i_ref_sas) * 100.0

    fig = plt.figure(figsize=(7.16, 6.2), dpi=dpi)
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

    vmax_int = 1.05
    vmax_err = 0.15

    # Panel (a): Standard ASM Intensity (Fixed Grid)
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(i_asm_disp, extent=extent_asm, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_a.set_title("(a) Standard ASM Intensity (Fixed Grid, $z = 0.80$ m)", fontsize=8)
    ax_a.set_xlabel("$x$ [mm]")
    ax_a.set_ylabel("$y$ [mm]")

    # Panel (b): Scalable Angular Spectrum Intensity (Scaled Grid)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(i_sas_disp, extent=extent_sas, origin="lower", cmap="inferno", vmin=0, vmax=vmax_int)
    ax_b.set_title("(b) Scalable Angular Spectrum Intensity (Scaled Grid)", fontsize=8)
    ax_b.set_xlabel("$x$ [mm]")
    ax_b.set_ylabel("$y$ [mm]")

    # Panel (c): Standard ASM Error Map
    ax_c = fig.add_subplot(gs[1, 0])
    im_c = ax_c.imshow(err_asm, extent=extent_asm, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_c.set_title("(c) Standard ASM Error Map (Aliasing)", fontsize=8)
    ax_c.set_xlabel("$x$ [mm]")
    ax_c.set_ylabel("$y$ [mm]")
    ax_c.text(
        0.04, 0.92, f"Relative $L_2$ error: {l2_asm:.2f}%",
        transform=ax_c.transAxes, fontsize=6.8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, lw=0)
    )

    # Panel (d): Scalable Angular Spectrum Error Map
    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(err_sas, extent=extent_sas, origin="lower", cmap="magma", vmin=0, vmax=vmax_err)
    ax_d.set_title("(d) Scalable Angular Spectrum Error Map", fontsize=8)
    ax_d.set_xlabel("$x$ [mm]")
    ax_d.set_ylabel("$y$ [mm]")
    ax_d.text(
        0.04, 0.92, f"Relative $L_2$ error: {l2_sas:.2f}%",
        transform=ax_d.transAxes, fontsize=6.8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, lw=0)
    )

    # Separate shared colorbars for top intensity row and bottom error row
    cbar_top = fig.colorbar(im_b, ax=[ax_a, ax_b], orientation="vertical", fraction=0.03, pad=0.04)
    cbar_top.set_label("Intensity $I / I_{\\mathrm{ref},\\max}$", fontsize=7)
    cbar_top.ax.tick_params(labelsize=6)

    cbar_bot = fig.colorbar(im_d, ax=[ax_c, ax_d], orientation="vertical", fraction=0.03, pad=0.04)
    cbar_bot.set_label("Absolute Error $|I - I_{\\mathrm{ref}}| / I_{\\mathrm{ref},\\max}$", fontsize=7)
    cbar_bot.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Far-Field Aliasing vs Scalable Angular Spectrum: Standard ASM vs SAS ($\\lambda = 532$ nm, $z = 0.80$ m)",
        fontsize=8.5, y=0.98
    )

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_fig03_results_profiles(
    output_png: str = "assets/fig03_results_profiles.png",
    output_pdf: str = "assets/fig03_results_profiles.pdf",
    dpi: int = 400,
) -> None:
    """Generate Figure 3: 2x2 grid combining 2D intensities with analytical profile plots."""
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
        f"First zero err: {r1_err:.2f}%\nSAS profile err: {prof_l2_err:.2f}%",
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
        "Analytical Fraunhofer Validation of Scalable Angular Spectrum ($z = 0.50$ m)",
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

    u_circ_in = circular_aperture(grid, radius=1.2e-4)
    u_rect_in = rectangular_aperture(grid, width=2.2e-4, height=3.2e-4)

    # Pre-render static top aperture images
    fig_top, (ax_ta, ax_tb) = plt.subplots(1, 2, figsize=(6.2, 2.5), dpi=120)
    fig_top.patch.set_facecolor("#111111")
    ax_ta.imshow(np.abs(u_circ_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_ta.set_title("Circular Aperture ($2a = 0.24$ mm)", fontsize=8, color="white")
    ax_ta.tick_params(colors="white")
    ax_ta.set_ylabel("$y$ [mm]", color="white")

    ax_tb.imshow(np.abs(u_rect_in), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax_tb.set_title("Rectangular ($0.22 \\times 0.32$ mm)", fontsize=8, color="white")
    ax_tb.tick_params(colors="white")
    plt.close(fig_top)

    z_min = 0.005
    z_max = 0.250
    z_forward = np.geomspace(z_min, z_max, num_steps)
    # Ping-pong sequence without duplicating endpoints for smooth loop
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

    # Far-field range where Fraunhofer analytical reference is rigorously valid
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

        l2_asm = relative_l2_error(i_asm, i_ref_asm) * 100.0
        l2_sas = relative_l2_error(i_sas, i_ref_sas) * 100.0

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
        ax_d.set_title(f"SAS Error (L2: {l2_sas:.1f}%)", fontsize=7.5, color="#66c2a5")
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
    print("Generating Primary diffraction loop GIF...")
    n_frames1 = generate_primary_loop_gif()
    print(f"Primary GIF complete ({n_frames1} frames).")
    print("Generating Error evolution loop GIF...")
    n_frames2 = generate_error_loop_gif()
    print(f"Error GIF complete ({n_frames2} frames).")
