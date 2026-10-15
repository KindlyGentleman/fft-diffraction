import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import scienceplots
    plt.style.use(["science", "ieee", "no-latex"])
except Exception:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
        "mathtext.fontset": "dejavuserif",
    })

METHOD_COLORS = {
    "fresnel": "#d95f02",
    "asm": "#7570b3",
    "blas": "#1b9e77",
    "reference": "#000000",
}

METHOD_LABELS = {
    "fresnel": "Fresnel (Paraxial)",
    "asm": "Standard ASM",
    "blas": "Band-Limited ASM",
    "reference": "Analytical Reference",
}

def save_multiformat(fig: plt.Figure, base_path: str, dpi: int = 300):
    """Save figure in PDF, SVG, and PNG formats."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_fig01_geometry(output_dir: str):
    """Figure 01: Schematic of optical diffraction geometry and coordinate axes."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
    ax.axis("off")

    # Draw incident wave
    for y_offset in np.linspace(-1.0, 1.0, 5):
        ax.annotate(
            "",
            xy=(-1.5, y_offset),
            xytext=(-2.5, y_offset),
            arrowprops=dict(arrowstyle="->", color="#386cb0", lw=1.5),
        )
    ax.text(-2.0, 1.25, "Incident Plane Wave\n($\\lambda = 532$ nm)", ha="center", fontsize=8, color="#386cb0")

    # Aperture plane
    ax.plot([-1.5, -1.5], [0.6, 1.5], color="black", lw=4)
    ax.plot([-1.5, -1.5], [-1.5, -0.6], color="black", lw=4)
    ax.plot([-1.5, -1.5], [-0.6, 0.6], color="#7fc97f", lw=1.5, ls="--")
    ax.text(-1.5, 1.6, "Aperture Plane\n$z = 0$", ha="center", fontsize=8, weight="bold")
    ax.text(-1.35, 0.0, "Aperture\n($2a$ or $W, H$)", ha="left", va="center", fontsize=7.5)

    # Observation plane
    ax.plot([1.8, 1.8], [-1.5, 1.5], color="#e7298a", lw=2.5)
    ax.text(1.8, 1.6, "Observation Plane\n$z = z_0$", ha="center", fontsize=8, weight="bold", color="#e7298a")

    # Rays and propagation distance
    ax.annotate(
        "",
        xy=(1.8, -1.7),
        xytext=(-1.5, -1.7),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(0.15, -1.95, "Propagation Distance $z$", ha="center", fontsize=8)

    ax.plot([-1.5, 1.8], [0.6, 1.2], color="#fdc086", ls=":", lw=1.2)
    ax.plot([-1.5, 1.8], [-0.6, -1.2], color="#fdc086", ls=":", lw=1.2)
    ax.plot([-1.5, 1.8], [0.0, 0.0], color="gray", ls="-.", lw=1.0)
    ax.text(1.9, 0.0, "Optical Axis ($z$)", va="center", fontsize=7.5, color="gray")

    ax.set_xlim(-2.8, 2.5)
    ax.set_ylim(-2.2, 2.0)
    save_multiformat(fig, os.path.join(output_dir, "fig01_geometry"))

def plot_fig02_apertures(grid, circ_field, rect_field, output_dir: str):
    """Figure 02: 2D transmission maps of circular and rectangular apertures."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=300)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    im0 = axes[0].imshow(np.abs(circ_field), extent=extent, origin="lower", cmap="viridis")
    axes[0].set_title("(a) Circular Aperture ($r = 0.15$ mm)")
    axes[0].set_xlabel("$x$ [mm]")
    axes[0].set_ylabel("$y$ [mm]")
    fig.colorbar(im0, ax=axes[0], label="Transmission $|U_0|$", fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.abs(rect_field), extent=extent, origin="lower", cmap="viridis")
    axes[1].set_title("(b) Rectangular ($0.30 \\times 0.45$ mm)")
    axes[1].set_xlabel("$x$ [mm]")
    axes[1].set_ylabel("$y$ [mm]")
    fig.colorbar(im1, ax=axes[1], label="Transmission $|U_0|$", fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig02_apertures"))

def plot_fig03_method_comparison(
    grid,
    i_fresnel,
    i_asm,
    i_blas,
    i_ref,
    z_val: float,
    output_dir: str,
):
    """Figure 03: Intensity and difference maps for Fresnel, ASM, and BL-ASM."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), dpi=300)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    vmax = float(np.max(i_blas))
    norm_kwargs = dict(extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)

    axes[0, 0].imshow(i_fresnel, **norm_kwargs)
    axes[0, 0].set_title("Fresnel Transfer Function")
    axes[0, 0].set_ylabel("$y$ [mm]")

    axes[0, 1].imshow(i_asm, **norm_kwargs)
    axes[0, 1].set_title("Conventional ASM")

    im_blas = axes[0, 2].imshow(i_blas, **norm_kwargs)
    axes[0, 2].set_title("Band-Limited ASM (BL-ASM)")
    fig.colorbar(im_blas, ax=axes[0, 2], label="Intensity [a.u.]", fraction=0.046, pad=0.04)

    err_fresnel = np.abs(i_fresnel - i_ref)
    err_asm = np.abs(i_asm - i_ref)
    err_blas = np.abs(i_blas - i_ref)
    err_vmax = max(np.max(err_fresnel), np.max(err_asm), np.max(err_blas))

    err_kwargs = dict(extent=extent, origin="lower", cmap="magma", vmin=0, vmax=err_vmax)
    axes[1, 0].imshow(err_fresnel, **err_kwargs)
    axes[1, 0].set_title(r"$|I_{\mathrm{Fresnel}} - I_{\mathrm{Ref}}|$")
    axes[1, 0].set_xlabel("$x$ [mm]")
    axes[1, 0].set_ylabel("$y$ [mm]")

    axes[1, 1].imshow(err_asm, **err_kwargs)
    axes[1, 1].set_title(r"$|I_{\mathrm{ASM}} - I_{\mathrm{Ref}}|$")
    axes[1, 1].set_xlabel("$x$ [mm]")

    im_err = axes[1, 2].imshow(err_blas, **err_kwargs)
    axes[1, 2].set_title(r"$|I_{\mathrm{BLAS}} - I_{\mathrm{Ref}}|$")
    axes[1, 2].set_xlabel("$x$ [mm]")
    fig.colorbar(im_err, ax=axes[1, 2], label="Absolute Error", fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_aspect("equal")

    fig.suptitle(f"Diffraction Field Comparison at $z = {z_val*1e3:.1f}$ mm", fontsize=10)
    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig03_method_comparison"))

def plot_fig04_analytical_validation(
    r_coords,
    prof_num,
    prof_ana,
    r1_theory,
    x_coords,
    slice_x_num,
    slice_x_ana,
    output_dir: str,
):
    """Figure 04: Analytical validation against Airy and rectangular Fraunhofer solutions."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), dpi=300)

    # Airy disc validation
    norm_num = prof_num / np.max(prof_num)
    norm_ana = prof_ana / np.max(prof_ana)
    r_mm = r_coords * 1e3
    r1_mm = r1_theory * 1e3

    axes[0].plot(r_mm, norm_ana, color="black", lw=1.5, label="Analytical Airy")
    axes[0].plot(r_mm, norm_num, color=METHOD_COLORS["blas"], ls="--", lw=1.5, label="Numerical BL-ASM")
    axes[0].axvline(r1_mm, color="red", ls=":", lw=1.0, label=f"First Zero ($r_1={r1_mm:.3f}$ mm)")
    axes[0].set_xlim(0, 3.0 * r1_mm)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Radius $r$ [mm]")
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_title("(a) Circular Aperture Airy Profile")
    axes[0].legend(loc="upper right", frameon=True)

    # Rectangular cross section
    x_mm = x_coords * 1e3
    norm_x_num = slice_x_num / np.max(slice_x_num)
    norm_x_ana = slice_x_ana / np.max(slice_x_ana)

    axes[1].plot(x_mm, norm_x_ana, color="black", lw=1.5, label="Analytical $\\mathrm{sinc}^2$")
    axes[1].plot(x_mm, norm_x_num, color=METHOD_COLORS["blas"], ls="--", lw=1.5, label="Numerical BL-ASM")
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("$x$ [mm]")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].set_title("(b) Rectangular Sinc Profile")
    axes[1].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig04_analytical_validation"))

def plot_fig05_sampling_convergence(df_sampling: pd.DataFrame, output_dir: str):
    """Figure 05: Convergence study under fixed FOV and fixed pixel pitch."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), dpi=300)

    # Fixed FOV
    df_fov = df_sampling[df_sampling["substudy"] == "fixed_fov"]
    for method in ["fresnel", "asm", "blas"]:
        sub = df_fov[df_fov["method"] == method]
        axes[0].plot(
            sub["N"],
            sub["l2_intensity_error"],
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            lw=1.2,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Grid Resolution $N$")
    axes[0].set_ylabel("Relative $L_2$ Intensity Error")
    axes[0].set_title("(a) Fixed Window Size ($S = 2.56$ mm)")
    axes[0].legend(loc="best", frameon=True)

    # Fixed Pitch
    df_pitch = df_sampling[df_sampling["substudy"] == "fixed_pitch"]
    for method in ["fresnel", "asm", "blas"]:
        sub = df_pitch[df_pitch["method"] == method]
        axes[1].plot(
            sub["N"],
            sub["l2_intensity_error"],
            marker="s",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            lw=1.2,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Grid Resolution $N$ (Expanding Window $S$)")
    axes[1].set_ylabel("Relative $L_2$ Intensity Error")
    axes[1].set_title("(b) Fixed Pixel Pitch ($\\Delta x = 5\\,\\mu$m)")
    axes[1].legend(loc="best", frameon=True)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig05_sampling_convergence"))

def plot_fig06_distance_study(df_distance: pd.DataFrame, output_dir: str):
    """Figure 06: Relative error vs propagation distance z with regime boundaries."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

    df_circ = df_distance[df_distance["aperture"] == "circular"]

    for method in ["fresnel", "asm", "blas"]:
        sub = df_circ[df_circ["method"] == method].sort_values("z_m")
        ax.plot(
            sub["z_m"] * 1e3,
            sub["l2_intensity_error"],
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            lw=1.4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Propagation Distance $z$ [mm]")
    ax.set_ylabel("Relative $L_2$ Error vs Reference")
    ax.set_title("Numerical Error vs Propagation Distance")
    ax.legend(loc="upper right", frameon=True)

    # Regime annotations
    ax.axvspan(1, 20, color="gray", alpha=0.15, label="Near / Fresnel Regime ($N_F > 1$)")
    ax.axvspan(100, 800, color="blue", alpha=0.08, label="Fraunhofer Regime ($N_F < 0.5$)")

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig06_distance_study"))

def plot_fig07_bandlimiting_analysis(
    grid_pad,
    h_asm,
    h_blas,
    mask,
    diff_map,
    z_val: float,
    output_dir: str,
):
    """Figure 07: Transfer function phase and band-limiting mask diagnostics."""
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.2), dpi=300)
    extent = [grid_pad.fx[0] * 1e-3, grid_pad.fx[-1] * 1e-3, grid_pad.fy[0] * 1e-3, grid_pad.fy[-1] * 1e-3]

    axes[0].imshow(np.angle(h_asm), extent=extent, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[0].set_title("(a) ASM Phase")
    axes[0].set_xlabel("$f_x$ [mm$^{-1}$]")
    axes[0].set_ylabel("$f_y$ [mm$^{-1}$]")

    axes[1].imshow(mask.astype(float), extent=extent, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) Band-Limit Mask")
    axes[1].set_xlabel("$f_x$ [mm$^{-1}$]")

    axes[2].imshow(np.angle(h_blas), extent=extent, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("(c) BL-ASM Phase")
    axes[2].set_xlabel("$f_x$ [mm$^{-1}$]")

    extent_sp = [grid_pad.x[0] * 1e3 / 2, grid_pad.x[-1] * 1e3 / 2, grid_pad.y[0] * 1e3 / 2, grid_pad.y[-1] * 1e3 / 2]
    im_diff = axes[3].imshow(diff_map, extent=extent_sp, origin="lower", cmap="hot")
    axes[3].set_title(r"$|I_{\mathrm{ASM}} - I_{\mathrm{BLAS}}|$")
    axes[3].set_xlabel("$x$ [mm]")
    fig.colorbar(im_diff, ax=axes[3], label="Diff", fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig07_bandlimiting_analysis"))

def plot_fig08_cost_accuracy(df_cost: pd.DataFrame, output_dir: str):
    """Figure 08: Accuracy vs computational runtime Pareto frontier."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=300)

    for method in ["fresnel", "asm", "blas"]:
        sub = df_cost[df_cost["method"] == method].sort_values("median_runtime_s")
        ax.plot(
            sub["median_runtime_s"],
            sub["l2_intensity_error"],
            marker="D",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            lw=1.3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"$N={int(row['N'])}$",
                (row["median_runtime_s"], row["l2_intensity_error"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median Execution Time [s]")
    ax.set_ylabel("Relative $L_2$ Intensity Error")
    ax.set_title("Computational Cost vs Accuracy Trade-off")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig08_cost_accuracy"))

def plot_fig09_failure_case(
    grid,
    i_asm_aliased,
    i_blas_clean,
    z_val: float,
    output_dir: str,
):
    """Figure 09: Numerical failure case showing transfer function aliasing."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=300)
    extent = [grid.x[0] * 1e3, grid.x[-1] * 1e3, grid.y[0] * 1e3, grid.y[-1] * 1e3]

    vmax = float(max(np.max(i_asm_aliased), np.max(i_blas_clean)))

    axes[0].imshow(i_asm_aliased, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title(f"(a) Standard ASM (Aliased at $z={z_val*1e3:.0f}$ mm)")
    axes[0].set_xlabel("$x$ [mm]")
    axes[0].set_ylabel("$y$ [mm]")

    im1 = axes[1].imshow(i_blas_clean, extent=extent, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title(f"(b) Band-Limited ASM (Clean at $z={z_val*1e3:.0f}$ mm)")
    axes[1].set_xlabel("$x$ [mm]")
    fig.colorbar(im1, ax=axes[1], label="Intensity [a.u.]", fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_multiformat(fig, os.path.join(output_dir, "fig09_failure_case"))
