# Playing with FFT Diffraction

A small experiment comparing Fresnel, angular spectrum, and band-limited angular spectrum propagation.

![Aperture and diffraction evolution across distance](assets/diffraction_loop.gif)

## What I tried

Simulating how a light wave spreads through free space using Fourier optics sounds straightforward on paper. Under the scalar Helmholtz equation, each plane wave component acquires a phase shift determined by its spatial frequency. Taking an FFT of the aperture field, multiplying by the free-space transfer function, and taking an inverse FFT should give the field at any distance $z$.

In practice, discrete grids make this trickier than the textbook formulas suggest. The transfer function phase factor $\exp(i k_z z)$ exhibits quadratic chirp behavior. As the propagation distance $z$ grows, its local oscillation frequency accelerates. Once the phase oscillates faster than one cycle per two grid pixels, the discrete grid aliases. Moiré fringes and wraparound artifacts corrupt the result. I wanted to test where standard propagators fail and how well the band-limiting method from Matsushima and Shimobaba (2009) handles the problem.

## Apertures and propagated fields

I set up monochromatic illumination ($\lambda = 532$ nm) on two canonical apertures: a circular pinhole ($2a = 0.30$ mm) and a rectangular slit ($0.30 \times 0.45$ mm), sampled on a $512 \times 512$ grid with pitch $\Delta x = 5.0\,\mu$m (computational window $S = 2.56$ mm). Twofold zero-padding was applied to prevent circular convolution wrap-around.

![Input apertures and corresponding diffraction fields](assets/fig01_aperture_diffraction.png)

Propagating to $z = 0.50$ m places both apertures in the Fraunhofer regime. The circular aperture produces the characteristic central Airy disc surrounded by concentric dark and bright rings. The rectangular aperture produces an orthogonal pattern of sinc sidelobes.

## When numerical sampling goes wrong

The standard Angular Spectrum Method (ASM) uses the exact free-space dispersion relation:

$$H_{\mathrm{AS}}(f_x, f_y, z) = \exp\left(i 2\pi z \sqrt{\frac{1}{\lambda^2} - f_x^2 - f_y^2}\right)$$

In continuous math, this is exact. On a discrete computer, the phase gradient $\partial \phi / \partial f_x$ exceeds the Nyquist limit beyond a critical distance $z_c = S \Delta x / \lambda$. For our setup ($S = 2.56$ mm, $\Delta x = 5\,\mu$m, $\lambda = 532$ nm), this cutoff occurs at just $z_c \approx 24$ mm.

When evaluated far beyond this threshold at $z = 0.80$ m, standard ASM fills the observation window with severe moiré interference artifacts:

![Standard ASM versus Band-Limited ASM with error maps](assets/fig02_methods_errors.png)

Band-Limited ASM (BL-ASM) addresses this by pruning spatial frequencies where the transfer function cannot be sampled properly:

$$f_{x,\mathrm{limit}} = \frac{1}{\lambda \sqrt{1 + \left(\frac{2z}{S_x}\right)^2}}$$

At $z = 0.80$ m, standard ASM produces 15.34% relative $L_2$ error (25.66% when comparing peak-normalized profiles). BL-ASM suppresses the unresolvable frequencies, dropping the error to 2.78% (a 5.5-fold improvement).

The animation below shows how this aliasing noise builds up in standard ASM as propagation distance increases, while BL-ASM remains clean throughout:

![Aliasing error evolution](assets/error_loop.gif)

## Does the simulation agree with theory?

To confirm that band-limiting does not destroy real physical information, I validated BL-ASM against analytical Fraunhofer solutions at $z = 0.50$ m:

![Analytical Fraunhofer validation](assets/fig03_results_profiles.png)

1. **Airy radial profile**: The measured first dark ring occurs at $r_1 = 1.088$ mm compared to the theoretical value of $1.081$ mm (0.61% relative position error). The overall radial profile matches the analytical $J_1(q)/q$ curve with 1.62% relative $L_2$ error.
2. **Rectangular sinc cross-sections**: Horizontal and vertical slices through the rectangular diffraction pattern match analytical $\mathrm{sinc}^2$ profiles with 1.65% and 2.64% relative error.
3. **Near-field direct integration**: At $z = 0.005$ m, where Fraunhofer formulas are invalid, BL-ASM agrees with an independent numerical Rayleigh-Sommerfeld direct integration within 3.89%.

## A few numbers

Summary of validation benchmarks and sampling convergence:

| Benchmark / Experiment | Configuration | Metric | Measured Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| Zero-distance identity | Circular, $z = 0.0$ m | Relative $L_2$ error | $2.0 \times 10^{-16}$ | Pass |
| Airy first dark ring | Circular, $z = 0.50$ m | Radius discrepancy | 0.61% | Pass |
| Airy radial profile | Circular, $z = 0.50$ m | Profile $L_2$ error | 1.62% | Pass |
| Sinc cross-section ($x$) | Rectangular, $z = 0.50$ m | Slice $L_2$ error | 1.65% | Pass |
| Sinc cross-section ($y$) | Rectangular, $z = 0.50$ m | Slice $L_2$ error | 2.64% | Pass |
| Rayleigh-Sommerfeld direct | Circular, $z = 0.005$ m | Near-field $L_2$ error | 3.89% | Pass |
| Power conservation | Circular, $z = 0.01$ m | Energy discrepancy | $< 0.03\%$ | Pass |

Sampling convergence at $z = 0.50$ m (fixed window $S = 2.56$ mm):

| Grid Points $N$ | Pitch $\Delta x$ | Fresnel Error | Standard ASM Error | BL-ASM Error |
| :--- | :--- | :--- | :--- | :--- |
| 256 | $10.0\,\mu$m | 6.61% | 6.79% | 2.95% |
| 512 | $5.0\,\mu$m | 7.33% | 7.11% | 1.87% |
| 1024 | $2.5\,\mu$m | 6.84% | 6.79% | 1.40% |

Under a fixed field of view, standard ASM and Fresnel saturate around 6.8% error because grid refinement cannot eliminate the transfer function phase chirp. BL-ASM error drops monotonically to 1.40% as grid resolution increases.

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

Run test suite:

```bash
python -m pytest -q
```

## References

- K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Optics Express* 17(22), 19662-19673 (2009). https://doi.org/10.1364/OE.17.019662
- waveprop: https://github.com/ebezzam/waveprop
- SciencePlots: https://github.com/garrettj403/SciencePlots
