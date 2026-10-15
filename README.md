# Numerical Accuracy and Sampling Limitations of FFT-Based Scalar Diffraction Models

[![Validation Suite](https://img.shields.io/badge/Validation%20Suite-8%2F8%20PASS-brightgreen.svg)](docs/VALIDATION_REPORT.md)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Program: Europhotonics](https://img.shields.io/badge/Academic%20Program-Europhotonics-purple.svg)](https://www.europhotonics.org/)

An independently reproducible scientific research project evaluating the numerical accuracy, sampling limits, and computational efficiency of FFT-based scalar diffraction algorithms in free space.

---

## 1. Research Overview

This repository implements, tests, and benchmarks three primary scalar wave propagation paradigms:
1. **Fresnel Transfer Function Method**: Paraxial quadratic-phase transfer function with fixed spatial sampling.
2. **Conventional Angular Spectrum Method (ASM)**: Exact non-paraxial free-space Helmholtz transfer function with evanescent mode attenuation.
3. **Band-Limited Angular Spectrum Method (BL-ASM)**: Sampling-constrained transfer function formulated by Matsushima and Shimobaba (2009) to suppress transfer function phase aliasing across near and far fields.

All numerical predictions are rigorously benchmarked against:
- Closed-form analytical Fraunhofer solutions (circular Airy pattern and rectangular sinc pattern).
- Non-FFT direct Rayleigh-Sommerfeld numerical integration.
- Reference implementation cross-checks (`waveprop` v0.0.12).

---

## 2. Key Scientific Findings

- **Transfer Function Aliasing Elimination**: In the far field ($z = 0.80$ m, $N_F = 0.053$), unbandlimited ASM suffers from severe local fringe frequency aliasing, elevating relative $L_2$ error to $15.33\%$. BL-ASM restricts the transfer function to the alias-free Nyquist band, keeping error below $2.78\%$ (a 5.5-fold error reduction).
- **Sub-Percent Analytical Feature Accuracy**: In the Fraunhofer regime ($z = 0.50$ m), BL-ASM predicts the theoretical Airy first dark ring radius ($r_1 = 1.081$ mm) within $0.61\%$ error without any empirical calibration parameters.
- **Resolution vs. Window Trade-Off**: Refining grid resolution ($N$) at a fixed physical window size cannot overcome spatial window truncation if extended diffraction features expand outside the window.
- **Computational Pareto Frontier**: BL-ASM adds less than $7\%$ execution runtime overhead compared to unbandlimited ASM while delivering a 4.8-fold error reduction at $N = 1024$.

---

## 3. Repository Architecture

```text
europhotonics-diffraction/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- CITATION.cff
|-- LICENSE
|-- .gitignore
|
|-- configs/
|   |-- quick.yaml
|   |-- full.yaml
|   |-- video.yaml
|   |-- validation.yaml
|
|-- src/
|   |-- diffraction/
|       |-- __init__.py
|       |-- grids.py
|       |-- apertures.py
|       |-- propagation.py
|       |-- bandlimit.py
|       |-- analytical.py
|       |-- metrics.py
|       |-- validation.py
|       |-- experiments.py
|       |-- plotting.py
|       |-- animation.py
|       |-- utils.py
|
|-- scripts/
|   |-- run_validation.py
|   |-- run_experiments.py
|   |-- generate_figures.py
|   |-- generate_video.py
|   |-- reproduce_all.py
|
|-- tests/
|   |-- test_grids.py
|   |-- test_apertures.py
|   |-- test_propagation.py
|   |-- test_analytical.py
|   |-- test_bandlimit.py
|   |-- test_energy.py
|   |-- test_convergence.py
|
|-- docs/
|   |-- PHYSICS_DERIVATION.md
|   |-- PAPER_REPRODUCTION_SCOPE.md
|   |-- REFERENCE_IMPLEMENTATION_AUDIT.md
|   |-- VALIDATION_REPORT.md
|   |-- NUMERICAL_LIMITATIONS.md
|   |-- REPRODUCIBILITY.md
|
|-- paper/
|   |-- MINI_PAPER.md
|   |-- RESULTS_TABLES.md
|   |-- FIGURE_CAPTIONS.md
|   |-- CLAIM_EVIDENCE_LEDGER.md
|
|-- outputs/
|   |-- figures/
|   |-- video/
|   |-- data/
|   |-- metrics/
```

---

## 4. Quick Start and Reproduction Commands

### Installation
Clone the repository and install dependencies in editable mode:
```bash
git clone https://github.com/europhotonics/diffraction-fft-accuracy.git
cd diffraction-fft-accuracy
pip install -e .
```

### Run Unit and Analytical Tests
Execute the full test suite with pytest:
```bash
python -m pytest -q
```
Expected output:
```text
15 passed in ~1.6s
```

### Run Scientific Validation Suite
Execute the automated validation suite:
```bash
python scripts/run_validation.py
```
Output:
```text
======================================================================
EUROPHOTONICS SCALAR DIFFRACTION - SCIENTIFIC VALIDATION SUITE
======================================================================
TEST NAME                              | STATUS   | KEY METRIC
----------------------------------------------------------------------
TEST_A_ZERO_DISTANCE                   | PASS     | BL-ASM err=2.77e-16
TEST_B_PLANE_WAVE                      | PASS     | BL-ASM err=0.00e+00
TEST_C_FFT_CONSISTENCY                 | PASS     | roundtrip err=3.86e-16
TEST_D_CIRCULAR_FRAUNHOFER             | PASS     | r1 err=0.61%, prof err=1.62%
TEST_E_RECTANGULAR_FRAUNHOFER          | PASS     | slice x err=1.65%, y err=2.64%
TEST_F_NEAR_FIELD_DIRECT_INTEGRATION   | PASS     | RS direct int err=3.89%
TEST_H_ENERGY_CONSERVATION             | PASS     | energy disc=0.02%
TEST_I_WAVEPROP_COMPARISON             | PASS     | waveprop L2 err=0.0000%
----------------------------------------------------------------------
OVERALL VALIDATION STATUS: ALL TESTS PASSED
```

### Run Experiments
Execute parameter sweeps across propagation distances, resolutions, and band-limiting conditions:
```bash
# Quick mode (~30 seconds on CPU)
python scripts/run_experiments.py --mode quick

# Comprehensive high-resolution mode
python scripts/run_experiments.py --mode full
```

### Generate IEEE Publication Figures
Regenerate all 9 publication figures in PDF, SVG, and PNG format:
```bash
python scripts/generate_figures.py
```

### Generate Educational Video Showcase
Render the animated video showcase:
```bash
# Rapid draft preview (15 s, 128x128 grid)
python scripts/generate_video.py --mode draft

# Full 1080p presentation (77 s, 1920x1080, 24 FPS)
python scripts/generate_video.py --mode full
```

### End-to-End One-Command Pipeline
Run the entire validation, experiment, figure, and video pipeline automatically:
```bash
python scripts/reproduce_all.py
```

---

## 5. Primary References

1. K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields," *Optics Express* 17(22), 19662–19673 (2009). DOI: [10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662).
2. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed. (Roberts and Company Publishers, 2005).
3. F. Shen and A. Wang, "Fast-Fourier-transform based numerical integration method for the Rayleigh–Sommerfeld diffraction formula," *Applied Optics* 45(6), 1102–1110 (2006).
4. E. Bezzam, `waveprop`: Numerical Simulation of Optical Wave Propagation in Python (2022).
