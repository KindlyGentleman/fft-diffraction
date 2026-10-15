# Reference Implementation Audit: `waveprop`

## 1. Reference Implementation Identification
- **Package**: `waveprop` (PyPI version 0.0.12)
- **Author**: Eric Bezzam
- **Repository**: https://github.com/ebezzam/waveprop
- **Audited Modules**:
  - `waveprop.rs`: Rayleigh-Sommerfeld, Angular Spectrum, and Band-Limited Angular Spectrum
  - `waveprop.fresnel`: Fresnel convolution and single-step propagators
  - `waveprop.fraunhofer`: Analytical circular and rectangular Fraunhofer profiles
  - `waveprop.util`: Sampling, padding, and coordinate generation

## 2. Key Findings and Mathematical Audit

### 2.1 Band-Limited Angular Spectrum Implementation (`waveprop.rs._bandpass`)
In `waveprop.rs._bandpass`, the frequency limit is computed using the shifted formulation from Matsushima (2010):
```python
du = 1 / (Sx)
u_limit_p = ((x0 + 1 / (2 * du)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
u_limit_n = ((x0 - 1 / (2 * du)) ** (-2) * z0**2 + 1) ** (-1 / 2) / wv
```
For on-axis propagation ($x_0 = 0$, $y_0 = 0$):
$$\frac{1}{2 \Delta u} = \frac{S_x}{2}$$
$$u_{\text{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2 z}{S_x}\right)^2}}$$
This strictly matches Equation (13) of Matsushima and Shimobaba (2009).

**Discovered Nuances in `waveprop`**:
1. **Asymmetric Boundary Inclusion**: In `waveprop.rs._bandpass`:
   ```python
   H_filter = (np.abs(fX - u0) <= fx_max) * (np.abs(fY - v0) < fy_max)
   ```
   Notice that the x-mask uses `<=` while the y-mask uses `<`. Our implementation will use consistent symmetric inclusive boundaries (`<=`).
2. **Zero-Distance Handling**: At $z = 0$, the mathematical limit of $u_{\text{limit}}$ is $1/\lambda$. In our implementation, we add an explicit bypass when $z = 0$ to return $H = 1$ identically, preventing floating-point division subtleties.

### 2.2 Fresnel Transfer Function Formulation (`waveprop.fresnel.fresnel_conv`)
In `waveprop.fresnel.fresnel_conv` (for equal sampling $m = d2/d1 = 1$):
```python
Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m / k * fsq)
u_out = ift2(Q2 * ft2(u_in, delta=d1), delta_f=df1)
```
Notice:
1. $2\pi^2 z / k = \pi \lambda z$, so $Q2 = \exp(-i \pi \lambda z (f_x^2 + f_y^2))$.
2. **Global Phase Factor**: `waveprop` omits the overall constant propagation phase $\exp(i k z)$. For intensity calculations ($|U|^2$), this omission is harmless because $|\exp(i k z)|^2 = 1$. However, for complex field comparisons with ASM or analytical formulas, omitting $\exp(i k z)$ introduces a global phase offset of $k z \pmod{2\pi}$. Our implementation explicitly includes $\exp(i k z)$ in the Fresnel transfer function:
   $$H_{\text{F}}(f_x, f_y; z) = \exp(i k z) \exp\left[ -i \pi \lambda z (f_x^2 + f_y^2) \right]$$

### 2.3 Evanescent Wave Treatment
In `waveprop.rs._form_transfer_function`:
```python
prop_waves = fsq <= 1 / wv_sq
evanescent_waves = np.logical_not(prop_waves)
H[prop_waves] = np.exp(1j * k * dz * np.sqrt(1 - wv_sq * fsq[prop_waves]))
H[evanescent_waves] = np.exp(-k * dz * np.sqrt(wv_sq * fsq[evanescent_waves] - 1))
```
Evanescent waves are correctly damped with $\exp(-k z \sqrt{\dots})$ for positive $z$. Our implementation preserves this correct causal decay and adds numerical floor guards against underflow warnings.

### 2.4 Analytical Fraunhofer Formulation (`waveprop.fraunhofer`)
In `waveprop.fraunhofer`:
- `jinc(x) = 2.0 * j1(np.pi * x) / (np.pi * x)` with proper $x \to 0$ handling ($1.0$).
- Analytical Airy:
  $$U_{\text{Airy}}(r) = \frac{\exp(i k z) \exp\left(i \frac{k r^2}{2z}\right)}{i \lambda z} \pi a^2 \left[ \frac{2 J_1(k a r / z)}{k a r / z} \right]$$
- Rectangular Fraunhofer:
  $$U_{\text{rect}}(x, y) = \frac{\exp(i k z) \exp\left(i \frac{k(x^2+y^2)}{2z}\right)}{i \lambda z} W H \operatorname{sinc}\left(\frac{W x}{\lambda z}\right) \operatorname{sinc}\left(\frac{H y}{\lambda z}\right)$$
Our independent analytical module implements these exact formulas with identical phase conventions.

## 3. Independent Cross-Check Plan
In our validation suite (Test I), we will run matched physical simulations through both our standalone engine and `waveprop.rs.angular_spectrum` and `waveprop.fresnel.fresnel_conv`, verifying relative agreement and documenting any minor phase or border-pixel differences.
