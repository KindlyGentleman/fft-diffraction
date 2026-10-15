# Playing with FFT Diffraction

A small numerical optics experiment exploring Fresnel, angular spectrum, band-limited, and scalable angular spectrum propagation in Python.

![Aperture and diffraction evolution across distance](assets/diffraction_loop.gif)

## What I tried

Simulating how a light wave spreads through free space using Fourier optics sounds straightforward on paper. Under the scalar Helmholtz equation, each plane wave component acquires a phase shift determined by its spatial frequency. Taking an FFT of the aperture field, multiplying by the free-space transfer function, and taking an inverse FFT should give the complex field at any distance $z$.

In practice, discrete grids make this trickier than the textbook formulas suggest. The transfer function phase factor $\exp(i k_z z)$ exhibits quadratic chirp behavior. As the propagation distance $z$ grows, its local oscillation frequency accelerates. Once the phase oscillates faster than one cycle per two grid pixels, the discrete grid aliases. Moiré fringes and wraparound artifacts corrupt the result.

Band-limited angular spectrum (Matsushima and Shimobaba, 2009) solves aliasing by pruning unresolvable high frequencies, but it keeps the destination grid locked to the input sampling pitch ($\Delta x_d = \Delta x_s$). When a diffracted beam expands in the far field, fixed-pitch grids either clip the pattern or require massive zero-padding. I wanted to see if Scalable Angular Spectrum propagation (Heintzmann, Lötgering, and Wechsler, 2023) could overcome this limitation without sacrificing non-paraxial accuracy or blowing up array sizes.

## Apertures and propagated fields

I set up monochromatic illumination ($\lambda = 532$ nm) on two canonical apertures: a circular pinhole ($2a = 0.30$ mm) and a rectangular slit ($0.30 \times 0.45$ mm), sampled on a $512 \times 512$ grid with pitch $\Delta x = 5.0\,\mu$m (computational window $S = 2.56$ mm). Twofold zero-padding was applied to prevent circular convolution wrap-around.

![Input apertures and corresponding diffraction fields](assets/fig01_aperture_diffraction.png)

Propagating to $z = 0.50$ m places both apertures in the Fraunhofer regime. The circular aperture produces the characteristic central Airy disc surrounded by concentric dark and bright rings. The rectangular aperture produces an orthogonal pattern of sinc sidelobes.

## Fixed grids versus scalable coordinates

The standard Angular Spectrum Method (ASM) evaluates the exact transfer function:

$$H_{\mathrm{AS}}(f_x, f_y, z) = \exp\left(i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)$$

On a discrete grid, the phase gradient $\partial \phi / \partial f_x$ exceeds the Nyquist limit beyond a critical distance $z_c = S \Delta x / \lambda$. For our setup ($S = 2.56$ mm, $\Delta x = 5\,\mu$m, $\lambda = 532$ nm), this cutoff occurs at just $z_c \approx 24$ mm. Beyond $z_c$, standard ASM fills the observation window with severe moiré interference artifacts.

Scalable Angular Spectrum (SAS) resolves this by decomposing the transfer function into a non-paraxial transfer difference precompensation followed by a single-step Fresnel Fourier transform:

$$U(x_d, y_d, z) = \frac{\exp(i k z)}{i \lambda z} \exp\left(\frac{i k}{2z}(x_d^2 + y_d^2)\right) \mathcal{F}\left\{ \mathcal{F}^{-1}\left[ \tilde{U}_0(f_x, f_y) \cdot W(f_x, f_y) \exp\left(i k z \left(H_{\mathrm{AS}} - H_{\mathrm{Fr}}\right)\right) \right] \right\}$$

where $H_{\mathrm{Fr}} = 1 - \frac{\lambda^2}{2}(f_x^2 + f_y^2)$. Because the final step is a spatial Fourier transform, the observation coordinates naturally scale with distance:

$$\Delta x_d = \frac{\lambda z}{N_{\mathrm{pad}} \Delta x_s}$$

At $z = 0.80$ m, standard ASM produces 15.34% relative $L_2$ error (25.66% when comparing peak-normalized profiles) due to grid aliasing. SAS evaluates the expanding field on its naturally scaled $42.6$ mm window with only 1.28% raw error (0.59% peak error):

![Standard ASM versus Scalable Angular Spectrum with error maps](assets/fig02_methods_errors.png)

The animation below demonstrates how aliasing noise builds up in standard ASM as propagation distance increases, while SAS cleanly tracks the expanding field:

![Aliasing error evolution](assets/error_loop.gif)

## Does the simulation agree with theory?

To confirm that SAS agrees with physical diffraction theory, I validated its predictions against analytical Fraunhofer solutions at $z = 0.50$ m:

![Analytical Fraunhofer validation](assets/fig03_results_profiles.png)

1. **Airy radial profile**: The measured first dark ring occurs at $r_1 = 1.086$ mm compared to the theoretical value of $1.081$ mm (0.44% relative position error). The overall radial profile matches the analytical $J_1(q)/q$ curve across the scaled $26.6$ mm field of view with 0.36% relative $L_2$ error.
2. **Rectangular sinc cross-sections**: Horizontal and vertical slices through the rectangular diffraction pattern match analytical $\mathrm{sinc}^2$ profiles with 1.89% and 2.76% relative error.
3. **Near-field direct integration**: At $z = 0.005$ m, where Fraunhofer formulas are invalid, BL-ASM agrees with an independent numerical Rayleigh-Sommerfeld direct integration within 3.89%.

## A few numbers

Summary of validation benchmarks:

| Benchmark / Experiment | Method | Metric | Measured Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| Zero-distance identity | All methods, $z = 0.0$ m | Relative $L_2$ error | $< 2.1 \times 10^{-16}$ | Pass |
| Airy first dark ring | SAS, $z = 0.50$ m | Radius discrepancy | 0.44% | Pass |
| Airy radial profile | SAS, $z = 0.50$ m | Profile $L_2$ error | 0.36% | Pass |
| Airy radial profile | BL-ASM, $z = 0.50$ m | Profile $L_2$ error | 1.62% | Pass |
| Sinc cross-section ($x$) | SAS, $z = 0.50$ m | Slice $L_2$ error | 1.89% | Pass |
| Sinc cross-section ($y$) | SAS, $z = 0.50$ m | Slice $L_2$ error | 2.76% | Pass |
| Rayleigh-Sommerfeld direct | BL-ASM, $z = 0.005$ m | Near-field $L_2$ error | 3.89% | Pass |
| Energy conservation | SAS, $z = 0.50$ m | Power ratio | $1.0000000$ | Pass |

Sampling convergence at $z = 0.50$ m (initial window $S = 2.56$ mm):

| Grid Points $N$ | Input Pitch $\Delta x_s$ | Fresnel Error | Standard ASM Error | BL-ASM Error | SAS Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 256 | $10.0\,\mu$m | 7.62% | 9.49% | 1.75% | 1.33% |
| 512 | $5.0\,\mu$m | 11.43% | 18.09% | 1.84% | 0.67% |
| 1024 | $2.5\,\mu$m | 13.08% | 13.63% | 1.89% | 0.40% |

Far-field aliasing comparison at $z = 0.80$ m ($N = 512, \Delta x_s = 5.0\,\mu$m):
- Standard ASM: 15.34% raw error (25.66% peak error)
- Single-step Fresnel: 15.28% raw error (31.99% peak error)
- Band-Limited ASM: 2.78% raw error (2.58% peak error)
- Scalable Angular Spectrum: 1.28% raw error (0.59% peak error)

## Regimes and trade-offs

SAS is not a universal replacement for all regimes:
- **Where SAS works best**: Intermediate and far-field distances ($z \ge 0.10$ m, magnification $m > 1$), where diffraction patterns expand beyond the source aperture window. It captures the entire beam on a naturally expanding grid without requiring 8x or 16x zero-padding arrays.
- **Where BL-ASM remains preferred**: Deep near-field propagation ($z \le 5$ mm, $m \ll 1$). In this regime, the SAS output grid spacing $\Delta x_d = \lambda z / (2L)$ becomes smaller than the input spacing, shrinking the field of view. Fixed-grid BL-ASM remains the natural choice for close-range diffraction.

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run fast verification:

```bash
python reproduce.py --quick
```

Run full experiment suite:

```bash
python reproduce.py
```

Run figure or GIF generation only:

```bash
python reproduce.py --figure-only
python reproduce.py --gif-only
```

Run test suite:

```bash
python -m pytest -q
```

## References

- R. Heintzmann, L. Lötgering, and F. Wechsler, "Scalable angular spectrum propagation," *Optica* 10(11), 1407-1416 (2023). https://doi.org/10.1364/OPTICA.499636
- K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Optics Express* 17(22), 19662-19673 (2009). https://doi.org/10.1364/OE.17.019662
- J. W. Goodman, *Introduction to Fourier Optics*, 4th ed. (W. H. Freeman, 2017).
