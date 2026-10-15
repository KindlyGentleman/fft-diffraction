# Paper Reproduction Scope and Benchmark Boundaries

## 1. Primary Scientific Citation
- **Paper**: K. Matsushima and T. Shimobaba, "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields," *Optics Express* 17(22), 19662–19673 (2009).
- **DOI**: [10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662)

## 2. Reproduction vs. Independent Benchmark Scope

This project is an **independent computational implementation and scientific validation study** inspired by the Matsushima-Shimobaba (2009) method. To preserve scientific honesty and reproducibility:

1. **Independent Implementation**: The algorithms in this codebase are implemented directly from first-principles wave equations and published analytical formulas, rather than being copied from existing code.
2. **Scope of Claims**:
   - We do not claim experimental optical laboratory validation; all results represent numerical wave optics simulations.
   - We do not claim an exact pixel-by-pixel reproduction of every arbitrary graphic in the 2009 paper, because the original paper used specific specialized test patterns (e.g., tilted planes, off-axis fields, computer-generated holograms) alongside idealized canonical apertures.
   - We reproduce the core mathematical foundation: the sampling frequency cutoff equations (Eq. 13 and Eq. 20 of Matsushima & Shimobaba 2009), the suppression of transfer function aliasing across the Fresnel and near-Fraunhofer regimes, and the comparative accuracy against conventional ASM, Fresnel, and analytical solutions.
3. **Research Questions Addressed**:
   - **RQ1 (Accuracy Comparison)**: Quantitative evaluation of Fresnel, standard ASM, and BL-ASM against analytical Airy and rectangular Fraunhofer solutions and direct Rayleigh-Sommerfeld integration.
   - **RQ2 (Distance Dependence)**: Error evolution across propagation distances from near field ($z = 1\text{ mm}$, Fresnel number $N_F \approx 42$) to intermediate ($z = 20\text{ mm}$, $N_F \approx 2.1$) and far field ($z = 800\text{ mm}$, $N_F \approx 0.05$).
   - **RQ3 (Sampling and Window Effects)**: Systematic separation of spatial sampling pitch $\Delta x$ and computational window size $S = N \Delta x$.
   - **RQ4 (Spectral Filtering Trade-Offs)**: Identifying regimes where BL-ASM removes aliasing artifacts versus regimes where excessive spectral clipping discards genuine diffraction detail.
   - **RQ5 (Computational Complexity vs. Error)**: Empirical execution time benchmarks and accuracy trade-offs.

## 3. Physical Parameters and Simulation Regimes

- **Wavelength**: $\lambda = 532\text{ nm}$ (frequency-doubled Nd:YAG / green laser).
- **Apertures**:
  - Circular aperture: radius $a = 0.15\text{ mm}$ ($D = 0.30\text{ mm}$).
  - Rectangular aperture: width $W = 0.30\text{ mm}$, height $H = 0.45\text{ mm}$.
- **Grid Configurations**:
  - Pilot grid: $N = 512 \times 512$, $\Delta x = \Delta y = 5\,\mu\text{m}$ (computational window $S = 2.56\text{ mm}$).
  - High-resolution grid: $N = 1024 \times 1024$, $\Delta x = \Delta y = 5\,\mu\text{m}$ ($S = 5.12\text{ mm}$).
  - Linear convolution zero-padding factor: $2\times$ (padded grid $2N \times 2N$).
- **Fresnel Number Analysis**:
  $$N_F = \frac{a^2}{\lambda z}$$
  - $z = 0.001\text{ m}$: $N_F \approx 42.3$ (Deep Fresnel / Near Field)
  - $z = 0.005\text{ m}$: $N_F \approx 8.46$ (Near Field)
  - $z = 0.020\text{ m}$: $N_F \approx 2.11$ (Intermediate Fresnel)
  - $z = 0.100\text{ m}$: $N_F \approx 0.42$ (Transition to Fraunhofer)
  - $z = 0.300\text{ m}$: $N_F \approx 0.14$ (Fraunhofer regime)
  - $z = 0.800\text{ m}$: $N_F \approx 0.053$ (Deep Fraunhofer regime)
