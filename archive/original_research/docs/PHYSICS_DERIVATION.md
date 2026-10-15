# Theoretical Foundations and Physics Derivations of Scalar Diffraction

## 1. Physical Assumptions and Scope

The diffraction models implemented in this repository are based on scalar diffraction theory in free space. The physical assumptions and their validity limits are explicitly delineated below:

1. **Scalar Approximation**: The optical electric field is treated as a single scalar complex amplitude $U(\mathbf{r})$, neglecting cross-polarization coupling and vectorial boundary conditions. This approximation is valid when the aperture dimensions $a$ and propagation distance $z$ are much larger than the optical wavelength $\lambda$ ($a \gg \lambda$, $z \gg \lambda$), and diffraction angles are moderate ($\theta \lesssim 30^\circ$).
2. **Monochromatic Coherent Illumination**: The field is assumed strictly time-harmonic with temporal angular frequency $\omega = 2\pi c / \lambda$, obeying a time dependence of $\exp(-i\omega t)$. The complex amplitude $U(\mathbf{r})$ represents the spatial component of the optical field.
3. **Homogeneous, Isotropic, Non-Dispersive Free Space**: The propagation medium has uniform refractive index $n = 1$, permittivity $\epsilon_0$, and permeability $\mu_0$, free from material dispersion, non-linearities, or volume scattering.
4. **Thin Aperture Approximation (Kirchhoff Boundary Condition)**: The diffracting screen is infinitesimally thin in the axial ($z$) direction. The field immediately behind the aperture plane ($z = 0^+$) is the product of the incident field $U_0(x, y)$ and the binary transmission function $t(x, y)$:
   $$U(x, y, 0^+) = U_0(x, y) t(x, y)$$
   Fringing fields and finite-conductivity edge currents are neglected.
5. **No Polarization or Vectorial Coupling**: Polarization-dependent transmission, polarization rotation at boundaries, and near-field subwavelength evanescent vector fields are outside the scalar model scope.

---

## 2. Derivation of the Angular Spectrum Method (ASM)

### 2.1 The Helmholtz Equation
In source-free homogeneous space, the scalar complex optical field satisfies the Helmholtz equation:
$$\nabla^2 U(x, y, z) + k^2 U(x, y, z) = 0$$
where $k = \frac{2\pi}{\lambda}$ is the optical wavenumber.

### 2.2 Two-Dimensional Spatial Fourier Transform
Let $\hat{U}(f_x, f_y; z)$ denote the 2D Fourier transform of $U(x, y, z)$ with respect to transverse spatial coordinates $(x, y)$:
$$\hat{U}(f_x, f_y; z) = \iint_{-\infty}^{\infty} U(x, y, z) \exp\left[-i 2\pi (f_x x + f_y y)\right] dx dy$$
The inverse Fourier transform reconstructs the spatial field:
$$U(x, y, z) = \iint_{-\infty}^{\infty} \hat{U}(f_x, f_y; z) \exp\left[i 2\pi (f_x x + f_y y)\right] df_x df_y$$

Substituting this inverse transform into the Helmholtz equation yields:
$$\iint_{-\infty}^{\infty} \left[ -(2\pi f_x)^2 - (2\pi f_y)^2 + \frac{\partial^2}{\partial z^2} + k^2 \right] \hat{U}(f_x, f_y; z) \exp\left[i 2\pi (f_x x + f_y y)\right] df_x df_y = 0$$

Since the exponential basis functions are orthogonal, the integrand must vanish for all $(f_x, f_y)$:
$$\frac{d^2 \hat{U}}{d z^2} + \left[ k^2 - (2\pi)^2 (f_x^2 + f_y^2) \right] \hat{U} = 0$$

Defining the longitudinal propagation constant $k_z$:
$$k_z = 2\pi \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}$$

The general solution for forward propagation ($z > 0$) without backward reflections is:
$$\hat{U}(f_x, f_y; z) = \hat{U}(f_x, f_y; 0) H_{\text{AS}}(f_x, f_y; z)$$
where $H_{\text{AS}}$ is the free-space Angular Spectrum transfer function:
$$H_{\text{AS}}(f_x, f_y; z) = \exp(i k_z z) = \exp\left( i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \right)$$

### 2.3 Propagating and Evanescent Wave Regimes
Depending on the spatial frequency magnitude $f_r^2 = f_x^2 + f_y^2$:
1. **Propagating Modes** ($f_x^2 + f_y^2 \le \frac{1}{\lambda^2}$):
   The radical is real, resulting in pure phase modulation:
   $$H_{\text{AS}}(f_x, f_y; z) = \exp\left( i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} \right), \quad |H_{\text{AS}}| = 1$$
2. **Evanescent Modes** ($f_x^2 + f_y^2 > \frac{1}{\lambda^2}$):
   The quantity under the square root is negative:
   $$\sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} = i \sqrt{f_x^2 + f_y^2 - \frac{1}{\lambda^2}}$$
   For forward propagation ($z > 0$), physical causality requires exponential decay:
   $$H_{\text{AS}}(f_x, f_y; z) = \exp\left( -2\pi z \sqrt{f_x^2 + f_y^2 - \frac{1}{\lambda^2}} \right)$$
   For $z \gg \lambda$, evanescent modes decay to numerical zero ($< 10^{-16}$) within a few wavelengths. In practical numerical implementations, unphysical exponential growth ($\exp(+|\alpha| z)$) must never occur.

---

## 3. Derivation of the Fresnel Transfer Function Method

When the transverse spatial frequencies satisfy the paraxial condition:
$$\lambda^2 (f_x^2 + f_y^2) \ll 1$$
we apply the Taylor series expansion to the square root:
$$\sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2} = \frac{1}{\lambda} \sqrt{1 - \lambda^2 (f_x^2 + f_y^2)} \approx \frac{1}{\lambda} \left[ 1 - \frac{\lambda^2}{2} (f_x^2 + f_y^2) \right] = \frac{1}{\lambda} - \frac{\lambda}{2} (f_x^2 + f_y^2)$$

Multiplying by $2\pi z$ gives the phase:
$$k_z z \approx \frac{2\pi z}{\lambda} - \pi \lambda z (f_x^2 + f_y^2) = k z - \pi \lambda z (f_x^2 + f_y^2)$$

Thus, the Fresnel transfer function is:
$$H_{\text{F}}(f_x, f_y; z) = \exp(i k z) \exp\left[ -i \pi \lambda z (f_x^2 + f_y^2) \right]$$

### 3.1 Properties of the Fresnel Transfer Function
1. At zero propagation distance ($z = 0$), $H_{\text{F}}(f_x, f_y; 0) = 1$, reproducing the input field identically.
2. The transfer function is quadratic in spatial frequencies and separable in $f_x$ and $f_y$:
   $$H_{\text{F}}(f_x, f_y; z) = \exp(i k z) \exp(-i \pi \lambda z f_x^2) \exp(-i \pi \lambda z f_y^2)$$
3. As spatial frequency increases, $H_{\text{F}}$ does not cut off; instead, its phase oscillates with frequency $\Omega_{f_x} = 2\pi \lambda z f_x$. At large distances or high frequencies, this causes numerical aliasing if the oscillation exceeds the Nyquist limit of the discrete frequency grid.

---

## 4. The Sampling Problem and the Band-Limited Angular Spectrum Method (BL-ASM)

### 4.1 Transfer Function Phase Oscillation and Local Spatial Frequency
In the conventional angular spectrum method, the transfer function phase is:
$$\phi(f_x, f_y) = 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}$$

Consider the 1D cross section along $f_x$ (with $f_y = 0$). The rate of phase change with respect to frequency $f_x$ defines the local spatial position (fringe frequency) in the spatial domain:
$$x_{\text{local}} = \frac{1}{2\pi} \frac{\partial \phi}{\partial f_x} = \frac{1}{2\pi} \frac{\partial}{\partial f_x} \left( 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2} \right) = -\frac{z f_x}{\sqrt{\frac{1}{\lambda^2} - f_x^2}}$$

### 4.2 Discrete Sampling and Nyquist Condition
In a discrete numerical computation with computational window length $S_x = N_x \Delta x$, the frequency sampling interval is:
$$\Delta f_x = \frac{1}{S_x} = \frac{1}{N_x \Delta x}$$
By the Nyquist-Shannon sampling theorem, the discrete sampling of $H_{\text{AS}}$ in the frequency domain is aliasing-free if and only if the local fringe frequency does not exceed half the sampling rate in frequency space:
$$\left| \frac{1}{2\pi} \frac{\partial \phi}{\partial f_x} \right| \le \frac{1}{2 \Delta f_x} = \frac{S_x}{2}$$

Substituting the expression for the local fringe frequency:
$$\frac{z |f_x|}{\sqrt{\frac{1}{\lambda^2} - f_x^2}} \le \frac{S_x}{2}$$

Squaring both sides:
$$\frac{z^2 f_x^2}{\frac{1}{\lambda^2} - f_x^2} \le \left(\frac{S_x}{2}\right)^2$$
$$z^2 f_x^2 \le \left(\frac{S_x}{2}\right)^2 \left( \frac{1}{\lambda^2} - f_x^2 \right) = \left(\frac{S_x}{2}\right)^2 \frac{1}{\lambda^2} - \left(\frac{S_x}{2}\right)^2 f_x^2$$
$$\left[ z^2 + \left(\frac{S_x}{2}\right)^2 \right] f_x^2 \le \left(\frac{S_x}{2}\right)^2 \frac{1}{\lambda^2}$$
$$f_x^2 \le \frac{\left(\frac{S_x}{2}\right)^2 \frac{1}{\lambda^2}}{z^2 + \left(\frac{S_x}{2}\right)^2} = \frac{1}{\lambda^2 \left[ 1 + \left(\frac{2 z}{S_x}\right)^2 \right]}$$

Taking the square root gives the Matsushima-Shimobaba critical band-limit frequency:
$$f_{x,\text{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 z}{S_x}\right)^2}}$$

Similarly, for the $y$-axis with window size $S_y = N_y \Delta y$:
$$f_{y,\text{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 z}{S_y}\right)^2}}$$

### 4.3 Formulation of the BL-ASM Transfer Function
The band-limited angular spectrum transfer function is defined as:
$$H_{\text{BLAS}}(f_x, f_y; z) = H_{\text{AS}}(f_x, f_y; z) \cdot M_{\text{BL}}(f_x, f_y)$$
where the band-limiting mask $M_{\text{BL}}$ is:
$$M_{\text{BL}}(f_x, f_y) = \begin{cases} 
1, & |f_x| \le f_{x,\text{limit}} \quad \text{and} \quad |f_y| \le f_{y,\text{limit}} \\
0, & \text{otherwise}
\end{cases}$$

### 4.4 Physical Significance of the Band-Limit
1. **Near-field limit ($z \to 0$)**:
   $$\lim_{z \to 0} f_{x,\text{limit}} = \frac{1}{\lambda}$$
   The band-limit naturally approaches the propagating wave boundary ($f_x \le 1/\lambda$), without restricting any physical propagating modes.
2. **Far-field limit ($z \gg S_x$)**:
   $$f_{x,\text{limit}} \approx \frac{S_x}{2 \lambda z}$$
   The maximum spatial frequency that can propagate to the observation window of size $S_x$ without folding back into the opposite edge via discrete circular convolution is bounded geometrically by rays originating at the aperture edges and ending at the observation window boundaries.
3. **Suppression of Wraparound Aliasing**:
   High spatial frequencies that would oscillate faster than $\Delta f_x$ produce spurious replica waves in the spatial domain due to discrete Fourier transform periodic boundary conditions. By clipping these components in $H_{\text{AS}}$, BL-ASM eliminates wraparound noise, at the trade-off of filtering out high-angle diffraction tails outside the computational window.

---

## 5. Direct Rayleigh-Sommerfeld Integration Reference

To independently validate the FFT-based propagators without relying on FFT approximations, we implement the First Rayleigh-Sommerfeld Diffraction Integral (RSI):
$$U(x, y, z) = -\frac{1}{2\pi} \iint_{\Sigma} U(x', y', 0) \frac{\partial}{\partial z} \left( \frac{\exp(i k r)}{r} \right) dx' dy'$$
where $r = \sqrt{(x - x')^2 + (y - y')^2 + z^2}$.

Carrying out the derivative with respect to $z$:
$$\frac{\partial}{\partial z} \left( \frac{\exp(i k r)}{r} \right) = \frac{z}{r} \frac{d}{dr} \left( \frac{\exp(i k r)}{r} \right) = \frac{z}{r} \left( \frac{i k \exp(i k r)}{r} - \frac{\exp(i k r)}{r^2} \right) = \frac{z \exp(i k r)}{r^2} \left( i k - \frac{1}{r} \right)$$

Therefore:
$$h_{\text{RS}}(x - x', y - y', z) = \frac{z}{2\pi r} \frac{\exp(i k r)}{r} \left( \frac{1}{r} - i k \right)$$

For optical regimes where $k r \gg 1$ ($r \gg \lambda / 2\pi$), the radiation term dominates:
$$h_{\text{RS}}(x - x', y - y', z) \approx \frac{z}{i \lambda r^2} \exp(i k r)$$

The continuous integral is discretized as a direct 2D Riemann sum over the aperture grid points $(x_m, y_n)$:
$$U(x_p, y_q, z) = \sum_{m} \sum_{n} U(x_m, y_n, 0) h_{\text{RS}}(x_p - x_m, y_q - y_n, z) \Delta x \Delta y$$

Because direct integration does not use FFTs or periodic boundary conditions, it is immune to discrete circular convolution and frequency aliasing. It serves as an independent benchmark on small grids.
