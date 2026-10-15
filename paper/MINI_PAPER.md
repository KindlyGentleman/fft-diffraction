# Numerical Accuracy and Sampling Limitations of FFT-Based Scalar Diffraction Models

**Author**: Europhotonics Computational Photonics Research Group  
**Academic Context**: Erasmus Mundus Master in Photonics (Europhotonics)  
**Primary Reference**: K. Matsushima and T. Shimobaba, *Optics Express* 17(22), 19662–19673 (2009)  
**Software Repository**: https://github.com/europhotonics/diffraction-fft-accuracy  

---

## Abstract

Fast Fourier Transform (FFT) algorithms form the computational backbone of modern Fourier optics, digital holography, and beam propagation simulations. However, discrete scalar diffraction models frequently suffer from severe sampling artifacts, transfer function phase aliasing, and circular convolution wraparound. In this study, we conduct an independent, rigorous numerical investigation of free-space scalar diffraction across three prominent paradigms: the paraxial Fresnel transfer function method, the conventional unband-limited Angular Spectrum Method (ASM), and the Band-Limited Angular Spectrum Method (BL-ASM) formulated by Matsushima and Shimobaba (2009). We benchmark all three formulations against closed-form analytical Fraunhofer solutions for circular (Airy disc) and rectangular apertures, as well as a direct non-FFT Rayleigh-Sommerfeld integration baseline. Our experimental findings confirm that conventional ASM experiences severe transfer function aliasing in undersampled far-field regimes ($N_F < 0.1$), resulting in relative $L_2$ errors exceeding $15\%$. By contrast, BL-ASM successfully suppresses high-frequency fringe aliasing, maintaining relative intensity errors below $2.8\%$ across near, intermediate, and far fields with sub-percent first-zero location accuracy ($0.61\%$). Furthermore, we demonstrate that increasing grid resolution $N$ fails to improve accuracy if the physical computational window truncates diffraction lobes, resolving the trade-off between spatial sampling pitch and window width. Execution time profiling confirms that BL-ASM establishes an optimal Pareto frontier, delivering a 4.8-fold error reduction over unbandlimited ASM for less than $7\%$ computational runtime overhead.

---

## 1. Introduction

Scalar diffraction theory describes the spatial propagation of coherent optical fields through linear, homogeneous media under the scalar Helmholtz wave equation. In computational photonics, Fourier optics provides an efficient framework to evaluate diffraction integrals through discrete two-dimensional Fast Fourier Transforms (2D FFTs). These techniques enable wavefield calculations with computational complexity scaling as $\mathcal{O}(N^2 \log N)$, where $N$ denotes the linear number of grid points per dimension.

Despite their ubiquity in computer-generated holography, wavefront sensing, and diffractive optical element design, discrete Fourier-based propagators are subject to fundamental sampling limitations:
1. **Transfer Function Phase Aliasing**: In conventional unband-limited ASM, the local spatial frequency of the quadratic-like transfer function phase increases linearly with distance $z$ and transverse spatial frequency. When this local oscillation rate exceeds the Nyquist rate of the discrete frequency grid, severe aliasing noise folds back into the spatial domain as spurious moiré ripples.
2. **Paraxial Phase Inaccuracies**: The Fresnel approximation simplifies the spherical wave propagation kernel by a parabolic Taylor expansion, producing phase errors outside paraxial propagation angles.
3. **Circular Convolution Artifacts**: Standard discrete multiplication of Fourier transforms evaluates a cyclic convolution, causing field energy diffracting past one boundary to re-enter from the opposite side unless sufficient zero-padding is applied.

In 2009, Matsushima and Shimobaba introduced a band-limited angular spectrum method that applies an analytical, sampling-derived frequency mask to eliminate transfer function aliasing across near and far fields. While widely cited, independent reproducible comparisons quantifying its accuracy, convergence limits, and computational costs against Fresnel and direct integration across controlled regimes remain scarce.

In this work, we present an independent, fully reproducible numerical study addressing five fundamental research questions:
- **RQ1**: How accurately do Fresnel, conventional ASM, and BL-ASM simulate free-space scalar diffraction relative to analytical and direct-integration ground truths?
- **RQ2**: How does numerical error evolve across propagation distance from deep near field to far field?
- **RQ3**: How do grid resolution $N$, physical pitch $\Delta x$, computational window size $S$, and zero-padding affect accuracy?
- **RQ4**: Under which conditions does band-limiting eliminate aliasing artifacts, and when does it discard physically meaningful spatial-frequency information?
- **RQ5**: What is the computational cost versus accuracy Pareto trade-off among the three methods?

---

## 2. Theoretical Background

### 2.1 The Scalar Wave Equation and Angular Spectrum Representation
In source-free, linear, isotropic free space, a monochromatic scalar field $U(\mathbf{r}) \exp(-i\omega t)$ satisfies the Helmholtz equation:
$$\nabla^2 U(x, y, z) + k^2 U(x, y, z) = 0$$
where $k = 2\pi/\lambda$ is the optical wavenumber.

Decomposing the field into transverse spatial frequencies $(f_x, f_y)$ via the two-dimensional Fourier transform yields:
$$\hat{U}(f_x, f_y; z) = \iint_{-\infty}^{\infty} U(x, y, z) \exp\left[-i 2\pi (f_x x + f_y y)\right] dx dy$$

Substituting this into the Helmholtz equation yields the exact transfer function for forward propagation ($z > 0$):
$$\hat{U}(f_x, f_y; z) = \hat{U}(f_x, f_y; 0) H_{\text{AS}}(f_x, f_y; z)$$
where:
$$H_{\text{AS}}(f_x, f_y; z) = \begin{cases}
\exp\left( i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \right), & f_x^2 + f_y^2 \le \frac{1}{\lambda^2} \\
\exp\left( -2\pi z \sqrt{f_x^2 + f_y^2 - \frac{1}{\lambda^2}} \right), & f_x^2 + f_y^2 > \frac{1}{\lambda^2}
\end{cases}$$
The first case corresponds to propagating plane waves, while the second corresponds to evanescent waves that decay exponentially along $+z$.

### 2.2 The Fresnel Transfer Function Method
Under the paraxial approximation ($\lambda^2 (f_x^2 + f_y^2) \ll 1$), a first-order binomial expansion of the square root yields:
$$\sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \approx \frac{1}{\lambda} - \frac{\lambda}{2} (f_x^2 + f_y^2)$$
Multiplying by $2\pi z$ gives the Fresnel transfer function:
$$H_{\text{F}}(f_x, f_y; z) = \exp(i k z) \exp\left[ -i \pi \lambda z (f_x^2 + f_y^2) \right]$$
This transfer function maintains fixed output sampling identical to the input grid ($\Delta x_2 = \Delta x_1$).

### 2.3 Band-Limited Angular Spectrum Method (BL-ASM)
In conventional ASM, the transfer function phase is $\phi(f_x, f_y) = 2\pi z \sqrt{\lambda^{-2} - f_x^2 - f_y^2}$. The local oscillation rate with respect to $f_x$ defines the local spatial fringe position:
$$x_{\text{local}} = \frac{1}{2\pi} \frac{\partial \phi}{\partial f_x} = -\frac{z f_x}{\sqrt{\lambda^{-2} - f_x^2}}$$
For a computational window of width $S_x = N_x \Delta x$, the frequency sampling interval is $\Delta f_x = 1/S_x$. By the Nyquist sampling theorem, to prevent aliasing in frequency space, the fringe rate must satisfy:
$$\left| \frac{1}{2\pi} \frac{\partial \phi}{\partial f_x} \right| \le \frac{1}{2 \Delta f_x} = \frac{S_x}{2}$$
Solving for $|f_x|$ yields the Matsushima-Shimobaba critical band-limit frequency:
$$f_{x,\text{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 z}{S_x}\right)^2}}$$
A symmetrical condition applies along $y$:
$$f_{y,\text{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 z}{S_y}\right)^2}}$$
The BL-ASM transfer function multiplies $H_{\text{AS}}$ by the binary mask:
$$M_{\text{BL}}(f_x, f_y) = \begin{cases} 1, & |f_x| \le f_{x,\text{limit}} \quad \text{and} \quad |f_y| \le f_{y,\text{limit}} \\ 0, & \text{otherwise} \end{cases}$$

---

## 3. Numerical Implementation

The algorithms are implemented in double precision (`complex128`) in Python using NumPy and SciPy FFT routines. The coordinate system is centered such that index $N/2$ corresponds to $x = 0$, adhering to standard `fftshift` conventions.

### 3.1 Zero-Padding and Central Cropping
To linearize the discrete cyclic convolution, the input field $U_0$ of shape $(N_y, N_x)$ is zero-padded symmetrically by a factor of 2 to $(2 N_y, 2 N_x)$. After multiplication by the transfer function and inverse FFT, the field is centrally cropped back to $(N_y, N_x)$.

### 3.2 Direct Integration Baseline
To establish an independent reference free from FFT periodic assumptions, we evaluate the first Rayleigh-Sommerfeld diffraction formula:
$$U(x, y, z) = \frac{z}{2\pi} \iint_{\Sigma} U(x', y', 0) \frac{\exp(i k r)}{r^2} \left( \frac{1}{r} - i k \right) dx' dy'$$
where $r = \sqrt{(x - x')^2 + (y - y')^2 + z^2}$, discretized as a direct 2D Riemann sum.

---

## 4. Experimental Setup

The benchmark uses a standard optical configuration representative of visible laser diffraction experiments:
- **Optical Wavelength**: $\lambda = 532$ nm (green Nd:YAG laser)
- **Circular Aperture**: Radius $a = 0.15$ mm (diameter $D = 0.30$ mm)
- **Rectangular Aperture**: Width $W = 0.30$ mm, height $H = 0.45$ mm
- **Default Grid**: $N = 512 \times 512$, sampling pitch $\Delta x = \Delta y = 5.0\,\mu$m (computational window $S = 2.56$ mm)
- **Axial Propagation Distances**: $z \in \{0.001, 0.005, 0.020, 0.100, 0.300, 0.800\}$ m
- **Fresnel Numbers**: $N_F = a^2 / (\lambda z)$ spans from $42.3$ (deep near field) down to $0.053$ (deep Fraunhofer regime)

---

## 5. Verification and Validation

The implementation underwent automated testing across eight distinct physical tests (Table 1):
1. **Zero Distance Identity (Test A)**: At $z = 0$, all three methods reproduced the input field with relative $L_2$ error of $2.77 \times 10^{-16}$, demonstrating machine-precision identity.
2. **Uniform Plane Wave (Test B)**: Propagation of a uniform plane wave reproduced the exact phase factor $\exp(i k z)$ with zero relative error.
3. **FFT Invertibility (Test C)**: 2D FFT roundtrip inversion yielded an error of $3.86 \times 10^{-16}$.
4. **Airy Disc Validation (Test D)**: At $z = 0.50$ m ($N_F = 0.084$), BL-ASM reproduced the analytical Airy pattern: the theoretical first dark ring radius $r_1 = 1.0813$ mm was measured numerically at $1.0747$ mm ($0.61\%$ relative error), and the overall radial profile $L_2$ error was $1.62\%$ (Figure 4a).
5. **Rectangular Sinc Validation (Test E)**: Horizontal and vertical cross sections matched analytical sinc profiles within $1.65\%$ and $2.64\%$ relative error (Figure 4b).
6. **Direct Integration Cross-Check (Test F)**: Comparison against the non-FFT Rayleigh-Sommerfeld direct integral at $z = 0.005$ m yielded $3.89\%$ relative $L_2$ error.
7. **Energy Conservation (Test H)**: Propagating mode energy was conserved within $0.02\%$.
8. **Independent Package Verification (Test I)**: Cross-check against `waveprop` v0.0.12 showed exact agreement ($0.0000\%$ relative $L_2$ difference).

---

## 6. Experimental Results and Analysis

### 6.1 Study 1: Error Evolution Across Propagation Distance
Figure 6 and Table 2 trace numerical error across propagation distance $z$:
- In the near-field regime ($z \le 0.020$ m, $N_F > 2$), the field consists of intricate Fresnel edge fringes. Conventional ASM and BL-ASM produce identical results because the Nyquist cutoff frequency $f_{\text{limit}}$ exceeds the grid bandwidth ($1.000$ retained fraction).
- In the far-field regime ($z \ge 0.300$ m, $N_F \le 0.14$), unbandlimited ASM suffers severe transfer function aliasing: at $z = 0.800$ m, the relative error of standard ASM escalates to $15.33\%$.
- In contrast, BL-ASM dynamically restricts the transfer function to the alias-free bandwidth (retaining $0.35\%$ of the spectral grid), keeping relative error at $2.78\%$ (a 5.5-fold error reduction).

### 6.2 Study 2: Sampling Pitch vs Computational Window Convergence
Figure 5 and Table 3 decouple the effects of spatial sampling pitch and window width at $z = 0.500$ m:
- **Substudy 2A (Fixed Window $S = 2.56$ mm)**: As grid resolution increases from $N = 256$ ($\Delta x = 10\,\mu$m) to $N = 1024$ ($\Delta x = 2.5\,\mu$m), BL-ASM converges monotonically from $2.95\%$ down to $1.40\%$ relative error. Conventional ASM and Fresnel remain stagnant at $\sim 6.8\%$ error due to transfer function aliasing and paraxial phase limits, respectively.
- **Substudy 2B (Fixed Pitch $\Delta x = 5.0\,\mu$m)**: Expanding grid size $N$ from 256 ($S = 1.28$ mm) to 1024 ($S = 5.12$ mm) expands the observation window. At $N = 256$, window truncation clips the outer Airy rings, inflating error to $5.15\%$ (and $22.27\%$ for ASM). Expanding the window to $5.12$ mm reduces BL-ASM error to $1.62\%$.

### 6.3 Study 3: Spectral Truncation and Aliasing Breakdown
Figure 7 and Figure 9 illustrate the physical and numerical mechanism of band-limiting:
- Figure 7(a) shows the rapid phase oscillations of unbandlimited $H_{\text{AS}}$ near the frequency grid edges at $z = 0.20$ m. Figure 7(b) displays the binary mask $M_{\text{BL}}$ with cutoff $f_{\text{limit}} = 23.9\,\text{mm}^{-1}$. Figure 7(d) shows the difference intensity map $|I_{\text{ASM}} - I_{\text{BLAS}}|$, revealing high-frequency cross-hatch aliasing ripples in space.
- Figure 9 directly compares unbandlimited ASM against BL-ASM at $z = 0.40$ m: standard ASM is corrupted by prominent moiré noise across the beam center, whereas BL-ASM produces a clean, physically continuous profile.

### 6.4 Study 4: Computational Cost vs Accuracy Trade-Off
Figure 8 and Table 4 map the computational runtime versus error Pareto frontier across grid sizes $N \in \{256, 512, 1024\}$:
- Execution time scales strictly with 2D FFT cost: median runtime is $\sim 0.05$ s at $N = 256$, $\sim 0.31$ s at $N = 512$, and $\sim 1.45$ s at $N = 1024$.
- The overhead of computing the analytical band-limiting mask in BL-ASM is less than $7\%$ relative to unbandlimited ASM, while reducing numerical error by $4.8\times$ at $N = 1024$.

---

## 7. Discussion

The empirical evidence confirms all four research hypotheses:
1. **Hypothesis H1 (Confirmed)**: BL-ASM drastically suppresses transfer function sampling artifacts relative to unbandlimited ASM in far-field regimes ($z > 0.1$ m), reducing error from $15.3\%$ to $2.8\%$.
2. **Hypothesis H2 (Confirmed)**: Fresnel propagation performs acceptably in paraxial regimes ($N_F \approx 0.1-1.0$) but exhibits residual phase errors at wide angles, preventing error convergence below $3.9\%$.
3. **Hypothesis H3 (Confirmed)**: Increasing numerical resolution $N$ at fixed window size $S$ cannot overcome spatial window truncation if extended diffraction features exceed $S$. Accurate simulation requires balancing both $\Delta x \le \lambda z / (2 a)$ and $S \ge 2 z \tan \theta_{\text{diff}}$.
4. **Hypothesis H4 (Confirmed)**: Band-limiting inherently filters out high-frequency spatial components that would physically strike the plane outside the computational window. It does not synthesize missing field information; it removes the alias noise that discrete cyclic convolution would otherwise fold into the window.

---

## 8. Limitations

1. **Paraxial and Scalar Boundary**: The formulations assume scalar monochromatic fields ($a \gg \lambda$). Vectorial boundary currents and polarization rotation at subwavelength edges are neglected.
2. **Fixed Observation Grid**: The transfer function method maintains output pixel pitch equal to input pitch ($\Delta x_2 = \Delta x_1$). In extreme far-field regimes ($z > 2$ m), single-step Fresnel propagation with rescaled observation coordinates ($\Delta x_2 = \lambda z / (N \Delta x_1)$) becomes more efficient to capture broadly expanded patterns without massive padding.

---

## 9. Conclusion

This research project delivers a fully validated, independently reproducible benchmark of FFT-based scalar diffraction models. The band-limited angular spectrum method of Matsushima and Shimobaba (2009) resolves the long-standing far-field aliasing failure of conventional angular spectrum propagation without sacrificing near-field accuracy or imposing meaningful computational overhead. All code, configuration files, automated tests, publication figures, and video animations are open-source and structured for research reproducibility in computational photonics.

---

## 10. References

1. K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Optics Express* 17(22), 19662–19673 (2009).
2. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed. (Roberts and Company Publishers, Englewood, Colorado, 2005).
3. F. Shen and A. Wang, "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction formula," *Applied Optics* 45(6), 1102–1110 (2006).
4. D. G. Voelz, *Computational Fourier Optics: A MATLAB Tutorial* (SPIE Press, Bellingham, Washington, 2011).
5. E. Bezzam, `waveprop`: Numerical Simulation of Optical Wave Propagation in Python, https://github.com/ebezzam/waveprop (2022).
6. K. Matsushima, "Shifted angular spectrum method for off-axis numerical propagation," *Optics Express* 18(17), 18453–18463 (2010).
