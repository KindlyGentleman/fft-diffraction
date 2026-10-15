# Numerical Limitations of FFT-Based Scalar Diffraction Models

This document outlines the fundamental physical and computational boundaries of discrete Fourier-based scalar wave propagation methods.

## 1. Physical Approximation Limits

1. **The Scalar Wave Barrier**:
   Scalar diffraction theory models only the transverse electric field scalar amplitude, neglecting vector boundary conditions and polarization coupling. When aperture features approach the optical wavelength ($a \lesssim 2\lambda$) or diffraction angles exceed approximately $30^\circ$ ($\text{NA} \gtrsim 0.5$), vectorial Maxwell solvers (such as FDTD or RCWA) are mandatory.
2. **Thin-Aperture (Kirchhoff) Assumption**:
   The aperture is treated as an infinitely thin binary planar mask. For realistic optical masks with finite thickness, sidewall roughness, or non-zero skin depth, volume conduction currents distort near-field transmission.
3. **Evanescent Wave Truncation**:
   Evanescent waves decay exponentially as $\exp(-2\pi z \sqrt{f_r^2 - \lambda^{-2}})$. For propagation distances $z > 5\lambda$, all evanescent components decay below $10^{-14}$. In this regime, scalar diffraction acts as a strict spatial low-pass filter with absolute cutoff at $f_r = 1/\lambda$.

---

## 2. Numerical Sampling and Discrete Fourier Transform Limits

### 2.1 The Transfer Function Aliasing Problem
The conventional Angular Spectrum transfer function phase is:
$$\phi(f_x, f_y) = 2\pi z \sqrt{\lambda^{-2} - f_x^2 - f_y^2}$$
Its local oscillation rate in spatial frequency space is:
$$\left| \frac{\partial (\phi / 2\pi)}{\partial f_x} \right| = \frac{z |f_x|}{\sqrt{\lambda^{-2} - f_x^2}}$$
In a discrete grid with window size $S_x = N_x \Delta x$, the frequency sampling interval is $\Delta f_x = 1/S_x$. By the Nyquist-Shannon sampling theorem, whenever the local fringe frequency exceeds $1/(2 \Delta f_x) = S_x / 2$, discrete sampling of the transfer function aliases.
- In the spatial domain, this frequency aliasing manifests as periodic wraparound: high-angle rays wrap across the computational boundary, creating artificial cross-hatch or moiré noise across the central pattern.
- This failure worsens at large propagation distances $z$, because the local fringe frequency scales linearly with $z$.

### 2.2 Band-Limiting: Mechanism and Trade-Off
The Matsushima-Shimobaba method restricts the transfer function to the alias-free frequency band:
$$|f_x| \le f_{x,\text{limit}} = \frac{1}{\lambda \sqrt{1 + (2 z / S_x)^2}}$$
- **Benefit**: Completely suppresses spurious high-frequency aliasing ripples, producing smooth, physically realistic diffraction profiles even in far-field regimes.
- **Trade-Off**: Band-limiting truncates genuine high-frequency diffraction components that would physically propagate outside the computational window of width $S_x$. In the far field ($z \gg S_x$), the retained spectral bandwidth shrinks inversely with distance: $f_{\text{limit}} \approx S_x / (2 \lambda z)$.
- Consequently, BL-ASM does not magically reconstruct diffraction features wider than the observation window; rather, it removes the aliased components that would otherwise contaminate the window.

### 2.3 Computational Window Truncation (Window Sizing vs. Grid Resolution)
A common misconception is that increasing grid points $N$ automatically increases simulation accuracy. As shown in Study 2:
- If $N$ is increased while the physical window size $S$ is held constant, the sampling pitch $\Delta x$ decreases, which refines spatial resolution but does not expand the observable diffraction pattern.
- If the physical diffraction pattern spreads beyond the window $S$, edge truncation errors arise regardless of how fine $\Delta x$ is.
- To simulate far-field diffraction accurately without coordinate rescaling, the computational window $S$ must be chosen large enough to enclose the outer diffraction rings of interest ($S \ge 2 z \tan \theta_{\text{diff}}$).

### 2.4 Linear Convolution and Zero-Padding
Discrete multiplication of spectra computes a circular convolution rather than a linear convolution. To linearize the discrete convolution, the aperture field must be zero-padded by a factor of at least 2 in each dimension prior to the FFT. Omitting zero-padding causes the field diffracting from one edge of the aperture to fold into the opposite edge.
