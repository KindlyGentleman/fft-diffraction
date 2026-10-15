# Figure Captions and Visual Metadata

## Figure 1: Simulation Geometry and Coordinate Definitions
**File**: `outputs/figures/fig01_geometry.{pdf,svg,png}`
**Caption**: Schematic representation of the scalar diffraction geometry. A monochromatic coherent plane wave ($\lambda = 532$ nm) is normally incident on an aperture plane at $z = 0$. The optical field diffracts through an aperture of characteristic width $2a$ and propagates along the optical axis $z$ to an observation plane located at $z = z_0$. The transverse coordinates $(x, y)$ are sampled on a discrete cartesian grid with pitch $\Delta x = \Delta y = 5.0\,\mu$m and total computational window width $S = N \Delta x$.

---

## Figure 2: Discrete Input Aperture Transmission Functions
**File**: `outputs/figures/fig02_apertures.{pdf,svg,png}`
**Caption**: Normalized amplitude transmission functions $|U_0(x, y)|$ for the canonical benchmark apertures sampled on a $512 \times 512$ grid with $5.0\,\mu$m spacing. (a) Circular aperture with radius $a = 0.15$ mm (diameter $D = 0.30$ mm). (b) Rectangular aperture with dimensions $W = 0.30$ mm and $H = 0.45$ mm.

---

## Figure 3: Method Comparison and Absolute Error Distributions
**File**: `outputs/figures/fig03_method_comparison.{pdf,svg,png}`
**Caption**: Spatial diffraction intensity and absolute error distributions at propagation distance $z = 50.0$ mm for a circular aperture ($a = 0.15$ mm, $\lambda = 532$ nm). Top row: 2D intensity distributions $|U(x, y)|^2$ computed using the Fresnel transfer function, conventional Angular Spectrum Method (ASM), and Band-Limited Angular Spectrum Method (BL-ASM) under identical numerical conditions ($N = 512$, padding factor 2). Bottom row: Absolute difference distributions $|I_{\text{method}} - I_{\text{ref}}|$ relative to the analytical Fraunhofer reference. All subpanels share the same spatial field of view and calibrated color scales.

---

## Figure 4: Analytical Validation Profiles
**File**: `outputs/figures/fig04_analytical_validation.{pdf,svg,png}`
**Caption**: Quantitative validation of numerical BL-ASM predictions against closed-form analytical solutions in the Fraunhofer regime at $z = 0.50$ m ($\lambda = 532$ nm). (a) Azimuthally averaged radial intensity profile for the circular aperture comparing numerical BL-ASM against the theoretical Airy disc function $I(r)/I(0) = [2 J_1(q)/q]^2$. The red vertical dotted line marks the analytical first dark ring radius $r_1 \approx 1.081$ mm ($0.61\%$ relative position discrepancy). (b) Central horizontal cross section ($y = 0$) for the rectangular aperture comparing numerical BL-ASM against the analytical $\operatorname{sinc}^2(W x / \lambda z)$ distribution ($1.65\%$ profile $L_2$ error).

---

## Figure 5: Grid Resolution and Computational Window Convergence
**File**: `outputs/figures/fig05_sampling_convergence.{pdf,svg,png}`
**Caption**: Numerical convergence of diffraction models at $z = 0.50$ m under two distinct sampling scaling regimes. (a) Fixed computational window size ($S = 2.56$ mm) with increasing resolution $N \in \{256, 512, 1024\}$ (decreasing sampling pitch $\Delta x \in \{10, 5, 2.5\}\,\mu$m). BL-ASM demonstrates monotonic convergence down to $1.40\%$ error, whereas unbandlimited ASM and Fresnel plateau due to transfer function aliasing and paraxial phase limits. (b) Fixed sampling pitch ($\Delta x = 5.0\,\mu$m) with increasing grid size $N \in \{256, 512, 1024\}$ (expanding window size $S = N \Delta x \in \{1.28, 2.56, 5.12\}$ mm), illustrating that window truncation at small $N$ severely distorts extended diffraction rings.

---

## Figure 6: Numerical Error Evolution Across Propagation Distance
**File**: `outputs/figures/fig06_distance_study.{pdf,svg,png}`
**Caption**: Relative $L_2$ error as a function of axial propagation distance $z \in [1\text{ mm}, 800\text{ mm}]$ for circular aperture diffraction ($\lambda = 532$ nm, $a = 0.15$ mm, $N = 512$, $\Delta x = 5\,\mu$m). The shaded gray band highlights the near-field Fresnel regime ($N_F > 1$), while the shaded blue band highlights the far-field Fraunhofer regime ($N_F < 0.5$). In the far field, unbandlimited ASM error rises sharply due to transfer function fringe frequency aliasing, whereas BL-ASM maintains low numerical error ($< 3\%$) by filtering under-sampled frequencies.

---

## Figure 7: Transfer Function Phase and Frequency Band-Limiting Analysis
**File**: `outputs/figures/fig07_bandlimiting_analysis.{pdf,svg,png}`
**Caption**: Diagnostic spectral representations of the Matsushima-Shimobaba band-limiting filter at $z = 0.20$ m on a padded $1024 \times 1024$ grid ($\Delta x = 5\,\mu$m). (a) Phase of the conventional ASM transfer function $\arg(H_{\text{AS}})$, showing rapid phase wrap oscillations near the grid perimeter. (b) Binary frequency mask $M_{\text{BL}}(f_x, f_y)$ defined by the analytical Nyquist cutoff frequencies $f_{x,\text{limit}} = f_{y,\text{limit}} \approx 23.9\,\text{mm}^{-1}$. (c) Filtered phase of the BL-ASM transfer function $\arg(H_{\text{BLAS}})$. (d) Resulting difference intensity map $|I_{\text{ASM}} - I_{\text{BLAS}}|$ in the spatial domain, isolating the high-frequency aliasing ripples suppressed by the band-limit.

---

## Figure 8: Accuracy versus Computational Runtime Pareto Frontier
**File**: `outputs/figures/fig08_cost_accuracy.{pdf,svg,png}`
**Caption**: Empirical accuracy versus execution runtime trade-off for Fresnel, standard ASM, and BL-ASM across grid sizes $N \in \{256, 512, 1024\}$. Benchmarks reflect median execution times over 3 repeats on an Intel x86-64 CPU under Python 3.13. BL-ASM achieves an optimal Pareto frontier, delivering a 4.8-fold error reduction over unbandlimited ASM at $N = 1024$ for negligible additional computational overhead ($< 7\%$ runtime difference for mask evaluation).

---

## Figure 9: Numerical Failure Case: Transfer Function Aliasing Breakdown
**File**: `outputs/figures/fig09_failure_case.{pdf,svg,png}`
**Caption**: Visual manifestation of transfer function sampling failure at $z = 0.40$ m. (a) Unbandlimited ASM displays a severe spurious moiré pattern and artificial intensity ripples caused by frequency folding across the Nyquist limit. (b) Band-Limited ASM correctly suppresses the under-sampled fringes, restoring a physically smooth diffraction envelope.
