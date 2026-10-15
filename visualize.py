"""Scientific visualization script for scalar diffraction experiments.

Generates:
1. Double-column 3x3 IEEE-style composite figure (PDF and 400+ DPI PNG).
2. Smoothly looping dual-panel animated GIF of field evolution.
"""

import os
from typing import Optional
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
    propagate_fresnel,
    airy_first_zero,
    airy_pattern,
    sinc_pattern,
    radial_profile,
    find_first_radial_minimum,
)

# IEEE style configuration
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
        "mathtext.fontset": "dejavuserif",
    })

COLORS = {
    "fresnel": "#d95f02",
    "asm": "#386cb0",
    "blas": "#1b9e77",
    "ref": "#222222",
}


def generate_composite_figure(
    output_png: str = "assets/diffraction_figure.png",
    output_pdf: str = "assets/diffraction_figure.pdf",
    dpi: int = 400,
) -> None:
    """Generate the complete 3x3 publication composite figure."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    fig = plt.figure(figsize=(7.16, 7.2), dpi=dpi)
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.38)

    # Simulation setup for 2D field maps
    nx, ny = 512, 512
    dx, dy = 5e-6, 5e-6
    wavelength = 532e-9
    radius = 1.5e-4
    z_map = 0.40

    grid = create_grid(nx, ny, dx, dy)
    u_circ = circular_aperture(grid, radius=radius)
    u_asm = propagate_asm(u_circ, wavelength, z=z_map, dx=dx, dy=dy, padding=2)
    u_blas = propagate_blas(u_circ, wavelength, z=z_map, dx=dx, dy=dy, padding=2)
    i_ref = airy_pattern(grid, wavelength, z=z_map, radius=radius)

    i_asm = np.abs(u_asm)**2
    i_blas = np.abs(u_blas)**2

    i_ref_norm = i_ref / np.max(i_ref)
    i_asm_norm = i_asm / np.max(i_asm)
    i_blas_norm = i_blas / np.max(i_blas)

    err_asm = np.abs(i_asm_norm - i_ref_norm)
    err_blas = np.abs(i_blas_norm - i_ref_norm)

    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    # Panel (a): Optical Diffraction Schematic
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    # Incident illumination arrows
    for y_pos in np.linspace(-0.8, 0.8, 5):
        ax_a.annotate(
            "",
            xy=(-1.2, y_pos),
            xytext=(-2.0, y_pos),
            arrowprops=dict(arrowstyle="->", color=COLORS["asm"], lw=1.2),
        )
    ax_a.text(-1.6, 1.1, "Incident\n$\\lambda = 532$ nm", ha="center", fontsize=6.5, color=COLORS["asm"])

    # Aperture screen
    ax_a.plot([-1.2, -1.2], [0.45, 1.3], color="black", lw=3.0)
    ax_a.plot([-1.2, -1.2], [-1.3, -0.45], color="black", lw=3.0)
    ax_a.plot([-1.2, -1.2], [-0.45, 0.45], color="#66a61e", lw=1.2, ls="--")
    ax_a.text(-1.2, 1.45, "Aperture\n$z = 0$", ha="center", fontsize=7, weight="bold")
    ax_a.text(-1.05, 0.0, "Diameter $2a$", ha="left", va="center", fontsize=6.5)

    # Observation screen
    ax_a.plot([1.4, 1.4], [-1.3, 1.3], color="#7570b3", lw=2.2)
    ax_a.text(1.4, 1.45, "Detector\n$z = z_0$", ha="center", fontsize=7, weight="bold", color="#7570b3")

    # Propagation ray annotations
    ax_a.plot([-1.2, 1.4], [0.45, 1.0], color="#e6ab02", ls=":", lw=1.0)
    ax_a.plot([-1.2, 1.4], [-0.45, -1.0], color="#e6ab02", ls=":", lw=1.0)
    ax_a.plot([-1.2, 1.4], [0.0, 0.0], color="gray", ls="-.", lw=0.8)
    ax_a.text(1.5, 0.0, "$z$", va="center", fontsize=6.5, color="gray")

    ax_a.annotate(
        "",
        xy=(1.4, -1.5),
        xytext=(-1.2, -1.5),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    ax_a.text(0.1, -1.75, "Distance $z$", ha="center", fontsize=6.5)
    ax_a.set_xlim(-2.2, 2.0)
    ax_a.set_ylim(-2.0, 1.8)
    ax_a.text(-2.1, 1.6, "(a)", fontsize=8, weight="bold")

    # Panel (b): Standard ASM Intensity (Aliased)
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(np.sqrt(i_asm_norm), extent=extent, origin="lower", cmap="magma", vmin=0, vmax=1)
    ax_b.set_title("Standard ASM ($z = 0.4$ m)", fontsize=7.5)
    ax_b.set_xlabel("$x$ [mm]")
    ax_b.set_ylabel("$y$ [mm]")
    ax_b.text(-1.15, 1.05, "(b)", fontsize=8, weight="bold", color="white")
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cbar_b.set_label("Norm. $|U|$", fontsize=6.5)
    cbar_b.ax.tick_params(labelsize=6)

    # Panel (c): BL-ASM Intensity (Clean)
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(np.sqrt(i_blas_norm), extent=extent, origin="lower", cmap="magma", vmin=0, vmax=1)
    ax_c.set_title("Band-Limited ASM ($z = 0.4$ m)", fontsize=7.5)
    ax_c.set_xlabel("$x$ [mm]")
    ax_c.set_ylabel("$y$ [mm]")
    ax_c.text(-1.15, 1.05, "(c)", fontsize=8, weight="bold", color="white")
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Norm. $|U|$", fontsize=6.5)
    cbar_c.ax.tick_params(labelsize=6)

    # Panel (d): Standard ASM Error Map
    ax_d = fig.add_subplot(gs[1, 0])
    vmax_err = 0.20
    im_d = ax_d.imshow(err_asm, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax_err)
    ax_d.set_title("ASM Error ($|I_{num} - I_{ref}|)$", fontsize=7.5)
    ax_d.set_xlabel("$x$ [mm]")
    ax_d.set_ylabel("$y$ [mm]")
    ax_d.text(-1.15, 1.05, "(d)", fontsize=8, weight="bold", color="white")
    cbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("Error", fontsize=6.5)
    cbar_d.ax.tick_params(labelsize=6)

    # Panel (e): BL-ASM Error Map (Shared color scale)
    ax_e = fig.add_subplot(gs[1, 1])
    im_e = ax_e.imshow(err_blas, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax_err)
    ax_e.set_title("BL-ASM Error ($|I_{num} - I_{ref}|)$", fontsize=7.5)
    ax_e.set_xlabel("$x$ [mm]")
    ax_e.set_ylabel("$y$ [mm]")
    ax_e.text(-1.15, 1.05, "(e)", fontsize=8, weight="bold", color="white")
    cbar_e = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
    cbar_e.set_label("Error", fontsize=6.5)
    cbar_e.ax.tick_params(labelsize=6)

    # Panel (f): Circular Airy Profile Validation at z = 0.50 m
    ax_f = fig.add_subplot(gs[1, 2])
    z_airy = 0.50
    u_airy_blas = propagate_blas(u_circ, wavelength, z=z_airy, dx=dx, dy=dy, padding=2)
    u_airy_asm = propagate_asm(u_circ, wavelength, z=z_airy, dx=dx, dy=dy, padding=2)
    i_airy_ref = airy_pattern(grid, wavelength, z=z_airy, radius=radius)

    r_num, prof_blas = radial_profile(np.abs(u_airy_blas)**2, grid.x, grid.y, num_bins=150)
    _, prof_asm = radial_profile(np.abs(u_airy_asm)**2, grid.x, grid.y, num_bins=150)
    r_ref, prof_ref = radial_profile(i_airy_ref, grid.x, grid.y, num_bins=150)

    r1_theory = airy_first_zero(wavelength, z_airy, radius) * 1e3
    r_mm = r_num * 1e3

    ax_f.plot(r_ref * 1e3, prof_ref / np.max(prof_ref), color=COLORS["ref"], ls="--", lw=1.2, label="Analytical")
    ax_f.plot(r_mm, prof_blas / np.max(prof_blas), color=COLORS["blas"], lw=1.0, label="BL-ASM")
    ax_f.plot(r_mm, prof_asm / np.max(prof_asm), color=COLORS["asm"], lw=0.8, ls=":", label="ASM")
    ax_f.axvline(r1_theory, color="#e41a1c", ls="-.", lw=0.8, label=f"$r_1={r1_theory:.2f}$ mm")

    ax_f.set_xlim(0, 1.25)
    ax_f.set_ylim(-0.02, 1.05)
    ax_f.set_title("Airy Radial Profile ($z=0.5$ m)", fontsize=7.5)
    ax_f.set_xlabel("Radius $r$ [mm]")
    ax_f.set_ylabel("Norm. Intensity")
    ax_f.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax_f.text(0.04, 0.96, "(f)", fontsize=8, weight="bold", transform=ax_f.transAxes)

    # Panel (g): Rectangular Aperture Sinc Cross-Sections at z = 0.50 m
    ax_g = fig.add_subplot(gs[2, 0])
    w_rect, h_rect = 3.0e-4, 4.5e-4
    u_rect = rectangular_aperture(grid, width=w_rect, height=h_rect)
    u_rect_out = propagate_blas(u_rect, wavelength, z=z_airy, dx=dx, dy=dy, padding=2)
    i_rect_num = np.abs(u_rect_out)**2
    i_rect_ana = sinc_pattern(grid, wavelength, z=z_airy, width=w_rect, height=h_rect)

    cx, cy = nx // 2, ny // 2
    x_mm = grid.x * 1e3
    y_mm = grid.y * 1e3

    ax_g.plot(x_mm, i_rect_ana[cy, :] / np.max(i_rect_ana[cy, :]), color=COLORS["ref"], ls="--", lw=1.1, label="Ana. $x$")
    ax_g.plot(x_mm, i_rect_num[cy, :] / np.max(i_rect_num[cy, :]), color=COLORS["blas"], lw=0.9, label="BL-ASM $x$")
    ax_g.plot(y_mm, i_rect_num[:, cx] / np.max(i_rect_num[:, cx]), color=COLORS["fresnel"], lw=0.9, ls="-.", label="BL-ASM $y$")

    ax_g.set_xlim(-1.2, 1.2)
    ax_g.set_ylim(-0.02, 1.05)
    ax_g.set_title("Rectangular Sinc ($z=0.5$ m)", fontsize=7.5)
    ax_g.set_xlabel("Coordinate [mm]")
    ax_g.set_ylabel("Norm. Intensity")
    ax_g.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax_g.text(0.04, 0.96, "(g)", fontsize=8, weight="bold", transform=ax_g.transAxes)

    # Panel (h): Fixed FOV Sampling Convergence (S = 2.56 mm)
    ax_h = fig.add_subplot(gs[2, 1])
    n_points = np.array([256, 512, 1024])
    # Audited numerical errors from study2_sampling.csv (fixed FOV)
    err_fov_fres = np.array([6.61, 7.33, 6.84])
    err_fov_asm = np.array([6.79, 7.11, 6.79])
    err_fov_blas = np.array([2.95, 1.87, 1.40])

    ax_h.plot(n_points, err_fov_fres, marker="o", color=COLORS["fresnel"], lw=1.0, ms=3.5, label="Fresnel")
    ax_h.plot(n_points, err_fov_asm, marker="s", color=COLORS["asm"], lw=1.0, ms=3.5, label="ASM")
    ax_h.plot(n_points, err_fov_blas, marker="^", color=COLORS["blas"], lw=1.2, ms=4.0, label="BL-ASM")

    ax_h.set_title("Fixed FOV ($S=2.56$ mm)", fontsize=7.5)
    ax_h.set_xlabel("Grid Points $N$")
    ax_h.set_ylabel("Relative Error [\\%]")
    ax_h.set_xticks(n_points)
    ax_h.set_ylim(0, 10)
    ax_h.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax_h.legend(loc="center right", frameon=False, fontsize=5.5)
    ax_h.text(0.04, 0.96, "(h)", fontsize=8, weight="bold", transform=ax_h.transAxes)

    # Panel (i): Fixed Pitch Window Expansion Convergence (dx = 5.0 um)
    ax_i = fig.add_subplot(gs[2, 2])
    # Audited numerical errors from study2_sampling.csv (fixed pitch, expanding window)
    fov_sizes = np.array([1.28, 2.56, 5.12])
    err_pitch_fres = np.array([22.07, 7.33, 3.90])
    err_pitch_asm = np.array([22.27, 7.11, 3.77])
    err_pitch_blas = np.array([5.15, 1.87, 1.62])

    ax_i.plot(fov_sizes, err_pitch_fres, marker="o", color=COLORS["fresnel"], lw=1.0, ms=3.5, label="Fresnel")
    ax_i.plot(fov_sizes, err_pitch_asm, marker="s", color=COLORS["asm"], lw=1.0, ms=3.5, label="ASM")
    ax_i.plot(fov_sizes, err_pitch_blas, marker="^", color=COLORS["blas"], lw=1.2, ms=4.0, label="BL-ASM")

    ax_i.set_title("Window Expansion ($\\Delta x=5\\,\\mu$m)", fontsize=7.5)
    ax_i.set_xlabel("Window Size $S$ [mm]")
    ax_i.set_ylabel("Relative Error [\\%]")
    ax_i.set_xticks(fov_sizes)
    ax_i.set_ylim(0, 25)
    ax_i.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax_i.legend(loc="upper right", frameon=False, fontsize=5.5)
    ax_i.text(0.04, 0.96, "(i)", fontsize=8, weight="bold", transform=ax_i.transAxes)

    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def generate_loop_gif(
    output_gif: str = "assets/diffraction_loop.gif",
    num_steps: int = 40,
    fps: int = 14,
) -> None:
    """Generate a compact, seamlessly looping side-by-side diffraction GIF."""
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    nx, ny = 256, 256
    dx, dy = 6e-6, 6e-6
    wavelength = 532e-9
    grid = create_grid(nx, ny, dx, dy)

    circ_in = circular_aperture(grid, radius=1.2e-4)
    rect_in = rectangular_aperture(grid, width=2.2e-4, height=3.2e-4)

    # Logarithmically spaced propagation distances from near to far field
    z_min = 0.005
    z_max = 0.250
    z_forward = np.geomspace(z_min, z_max, num_steps)
    # Forward and backward parameter sweep for seamless looping
    z_cycle = np.concatenate([z_forward, z_forward[::-1]])

    frames = []
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    for z_val in z_cycle:
        u_circ = propagate_blas(circ_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)
        u_rect = propagate_blas(rect_in, wavelength, z=z_val, dx=dx, dy=dy, padding=2)

        # Gamma compressed normalized amplitude for clear fringe visibility
        i_circ = np.abs(u_circ)**2
        i_rect = np.abs(u_rect)**2
        amp_circ = np.sqrt(i_circ / np.max(i_circ))
        amp_rect = np.sqrt(i_rect / np.max(i_rect))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.2), dpi=120)
        fig.patch.set_facecolor("#111111")

        ax1.imshow(amp_circ, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax1.set_title(f"Circular ($2a = 0.24$ mm)", fontsize=8, color="white")
        ax1.set_xlabel("$x$ [mm]", color="white")
        ax1.set_ylabel("$y$ [mm]", color="white")
        ax1.tick_params(colors="white")

        ax2.imshow(amp_rect, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax2.set_title(f"Rectangular ($0.22 \\times 0.32$ mm)", fontsize=8, color="white")
        ax2.set_xlabel("$x$ [mm]", color="white")
        ax2.tick_params(colors="white")

        # Global header with current distance and Fresnel number
        fresnel_num = (1.2e-4)**2 / (wavelength * z_val)
        fig.suptitle(f"$z = {z_val * 1e3:5.1f}$ mm   |   $N_F = {fresnel_num:4.2f}$", fontsize=9, color="#66c2a5", y=0.98)
        fig.tight_layout()

        # Render frame to memory buffer
        fig.canvas.draw()
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba_buffer).convert("P", palette=Image.ADAPTIVE)
        frames.append(img)
        plt.close(fig)

    # Save animated GIF with infinite loop
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    generate_composite_figure()
    generate_loop_gif()
