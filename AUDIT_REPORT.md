# Scientific Software Audit and Verification Report: Numerical FFT Diffraction

**Repository:** KindlyGentleman/fft-diffraction  
**Audited Commit Baselines:** `e59f39429d10ca6a15781e77da08bb3583ae9247` (First Round), `9527ceebb44610db69eea2ab839de2f5292940b1` (Second Round)  
**Execution Environment:** Windows 11 Pro, Intel Core i7-13700H (16 physical/logical threads), 15.73 GB RAM, Python 3.13.7, NumPy 2.5.3, SciPy 1.16.2  
**Date:** October 2026  

---

## 1. Executive Summary

A comprehensive two-round scientific audit and numerical verification was conducted on the `fft-diffraction` repository. The codebase models monochromatic scalar optical diffraction using four FFT-based propagators: Fresnel transfer function, standard Angular Spectrum Method (ASM), Band-Limited Angular Spectrum Method (BL-ASM) [R2], and Scalable Angular Spectrum (SAS) [R1], benchmarked against analytical solutions (Airy pattern, 2D rectangular sinc, Gaussian ABCD beam propagation) and direct numerical integration (Rayleigh–Sommerfeld).

### Summary of Audit Findings and Resolutions:
1. **Artifact Truncation & Unsafe Failure Path (Second-Round Critical Defect):** In commit `9527cee`, a newly added regression test injected an intentional validation failure against production file paths. In `reproduce.py`, the failure branch called `save_results(validations, [])` with an empty metrics list, triggering `IndexError: list index out of range` at `metrics[0].keys()`. This aborted execution after truncating `results/metrics.csv` to 0 bytes and overwriting `results/validation.json` with a single failure stub. The issue was resolved by adding `--output-dir` to `reproduce.py`, isolating all subprocess tests to pytest's `tmp_path`, making writes atomic via `.tmp` staging, adding safe fallback headers for empty metrics, and logging validation diagnostics to a separate `validation_failure.json` without overwriting validated results.
2. **SAS Magnification Condition and Dead-Code Vignetting Bug:** In `diffraction.py`, `check_sas_validity` contained `if denom > 0:`, which was dead code because the analytical denominator $w \cdot (-1 + 2\sqrt{2}\sqrt{\dots})$ is strictly negative for physical optical parameters; this silenced the vignetting calculation. Furthermore, the magnification condition was previously described as $M = \max(M_x, M_y) > 1$. The true physical condition requires both axes to expand without chirp aliasing ($M_x > 1$ and $M_y > 1$), yielding $M = \min(M_x, M_y) > 1$.
3. **Contaminated Method Selection Map (Issue A):** In the baseline implementation, when in-memory benchmark metrics were passed directly to `generate_fig04_method_selection_map`, mixed experiment records (`fixed_fov` alongside `method_selection_map`) overwrote grid cells and distorted the optimal method classification. Furthermore, color scaling was artificially clipped at $0.005$. Resolved by strict experiment filtering, deduplication asserting consistency, and dynamic error colormap scaling.
4. **Artificial Theoretical Substitution (Issue B):** Optical vortex validation substituted ideal integer values ($\ell$) whenever experimental charge extraction failed or yielded NaN. Resolved by completely removing theoretical fallbacks and testing signed charges including negative values ($\ell = -1$).
5. **Rounding Concealment and Resolution Delineation (Issue C):** Baseline thin-lens validation applied `np.round(..., 5)` and `np.round(..., 7)`, artificially masking numerical discrepancy. The unrounded floating-point measurements are now reported with explicit numerical uncertainties: Quick Mode ($N=256$, $\Delta x = 10\,\mu\text{m}$) yields $z_f = 73.5330$ mm ($0.0036\%$ error), while Full Mode ($N=512$, $\Delta x = 5\,\mu\text{m}$) yields $z_f = 73.5350$ mm ($0.0008\%$ error), demonstrating genuine spatial grid convergence towards theoretical $z_f = 73.5356$ mm.
6. **Asymmetric Coordinate Centering on Odd Grids (Defects 2.1 & 2.2):** Baseline grid creation used `n / 2.0` (float division) which misaligned with the integer FFT zero-frequency origin `n // 2`. Standardized coordinate center to `cx = nx // 2` and `pad_field` to `(new_ny // 2) - (ny // 2)`.
7. **Complex Field Metric Integration:** Analytical global phase alignment $\phi = -\operatorname{angle}(\sum U_{\mathrm{num}} U_{\mathrm{ref}}^*)$ was integrated into `compute_complex_field_error` and benchmarked against Rayleigh–Sommerfeld direct integration ($7.81\%$ complex field error, $-0.0034$ rad phase alignment).
8. **Automated CI Quality Gate:** CI workflow was hardened with `git diff --exit-code -- results/` ensuring tests never modify tracked results, followed by rigorous content validation of all 7 checks in `validation.json` and $> 20$ rows in `metrics.csv`.

---

## 2. Audit Ledger and Issue Resolution

| Issue ID | File | Lines (approx.) | Baseline / Previous Behavior | Root Cause | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Artifact Corruption** | `reproduce.py`, `test_diffraction.py` | 580–665 | `results/metrics.csv` truncated to 0 bytes; `validation.json` contained failing stub; CI passed in fresh container while repo was corrupted. | Test injected failure into production path; empty `metrics` list caused `IndexError` on `metrics[0]`. | **Fixed & Verified** | Added `--output-dir`, isolated tests to `tmp_path`, implemented atomic writes, handled empty metrics safely, wrote failures to `validation_failure.json`. |
| **SAS Vignetting & Magnification** | `diffraction.py` | 530–575 | Vignetting limits never calculated (`z_vignette_limits` always empty); magnification criterion inconsistently defined. | Guard `if denom > 0` was impossible since `denom < 0`; used $\max(M_x, M_y)$ instead of physical $\min$. | **Fixed & Verified** | Changed guard to `abs(denom) > 1e-18`; evaluated $z_{\mathrm{vig}} = \min(z_{\mathrm{vig},x}, z_{\mathrm{vig},y})$; enforced $M = \min(M_x, M_y) > 1$. |
| **Issue A** | `visualize.py` | 440–540 | Figure 4 mixed `fixed_fov` with `method_selection_map` when called in-memory; static LogNorm clipped at 0.005. | In-memory `metrics_data` passed without experiment filtering; hard-coded `vmin=0.005`. | **Fixed & Verified** | Strictly filtered `experiment == 'method_selection_map'`, added cell deduplication checking conflicting duplicates, dynamic `vmin_err`. |
| **Issue B** | `visualize.py`, `reproduce.py` | 850–960 | Vortex charge substituted theoretical `float(ell)` if measurement missing; bar plot crashed on $\ell < 0$. | Fallback `.get("charge_input_measured", float(ell))` and static `ylim(0, 3.8)`. | **Fixed & Verified** | Removed theoretical substitution; evaluated only genuine measurements; dynamic signed y-limits and tick formatting for $\ell = -1$. |
| **Issue C** | `reproduce.py` | 240–280, 640–660 | Failed validations logged error but exited with 0; lens focus rounded with `np.round(..., 5/7)` claiming $0.0000\%$ error. | Missing `sys.exit(1)`; artificial floating-point rounding. | **Fixed & Verified** | Added `sys.exit(1)` upon validation failure; evaluated unrounded floats ($N=256$: $0.0036\%$, $N=512$: $0.0008\%$) with $\pm dz/10$ uncertainty. |
| **Defect 2.1** | `diffraction.py` | 65–85 | Odd grid coordinates shifted by half-pixel relative to FFT origin. | Float division `cx = nx / 2.0` vs integer `nx // 2` used by `fftfreq`. | **Fixed & Verified** | Standardized coordinate center to `cx = nx // 2`, aligning spatial indices with `fftshift(fftfreq(n, d))`. |
| **Defect 2.2** | `diffraction.py` | 130–155 | Odd grid padding shifted aperture center by one pixel. | `sy = (new_ny - ny) // 2` did not match centering convention. | **Fixed & Verified** | Adjusted center offset to `sy = (new_ny // 2) - (ny // 2)` ensuring exact center pixel preservation. |
| **Defect 3.1** | `diffraction.py` | 740–790 | `compare_on_common_grid` assumed zero-symmetric bounds and zero-filled outer values. | Unchecked grid overlap and arbitrary fill values in `RegularGridInterpolator`. | **Fixed & Verified** | Extracted exact physical bounding box intersection $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$, asserting valid overlap before interpolation. |
| **Defect 3.2** | `diffraction.py` | 800–840 | Complex field error evaluation distorted by global constant phase offsets and background noise. | Direct subtraction without global phase alignment factor. | **Fixed & Verified** | Implemented analytical phase alignment via $\phi = -\operatorname{angle}(\sum U_{\mathrm{num}} U_{\mathrm{ref}}^*)$; amplitude threshold masking above noise floor. |

---

## 3. Detailed Technical Analysis

### 3.1 Artifact Corruption and Safe Failure Paths
In commit `9527cee`, `results/metrics.csv` was 0 bytes and `results/validation.json` contained only `{"failing_check": {"passed": false}}`. Investigation revealed:
1. `test_regression_issue_c_validation_failure_exit_status` invoked `python reproduce.py` without an output directory argument, defaulting to the production `results/` folder.
2. In `reproduce.py`, the validation failure branch called `save_results(validations, [])`.
3. Inside `save_results()`, `fieldnames = list(metrics[0].keys())` failed with `IndexError` because `metrics` was empty.
4. The CSV file was opened in write mode, truncated to 0 bytes, and left empty when the exception terminated the script.
5. In the repaired codebase:
   - `save_results()` stages writes to `.tmp` files and executes atomic `os.replace`.
   - Empty `metrics` lists fall back to standard `BENCHMARK_COLUMNS` header rather than raising `IndexError`.
   - A validation failure writes diagnostics to `validation_failure.json` and exits with status 1 without touching existing valid benchmark results.
   - Regression tests run with `--output-dir <tmp_path>`, completely isolating test executions from production data.

### 3.2 SAS Magnification and 2D Vignetting Physics
In the Scalable Angular Spectrum method (Heintzmann et al., 2023 [R1]):
- Observation coordinates scale with distance: $\Delta x_d = \lambda z / (p N_x \Delta x_s)$, defining magnification $M_x = \Delta x_d / \Delta x_s = \lambda z / (p N_x \Delta x_s^2)$.
- If $M \le 1$, the destination grid shrinks relative to the source grid, and the quadratic source phase chirp $\exp(i \pi (1 - M) r^2 / (\lambda z))$ exceeds the Nyquist limit, introducing chirp aliasing. Therefore, SAS is valid only when **both** axes expand ($M_x > 1$ and $M_y > 1$), i.e., $M = \min(M_x, M_y) > 1$.
- Vignetting occurs when the expanded output field exceeds the bandwidth of the intermediate transfer function (Eq. 18 of [R1]). In the previous code, `denom = wavelength * (-1.0 + 2.0 * np.sqrt(2.0) * term)` was checked with `if denom > 0:`. However, for physical optical parameters where sampling pitch $d \gg \lambda$, `term < 1/(2*sqrt(2))`, making `denom < 0`. Because `num` is also negative, `z_vig = num / denom > 0` is physically valid. Removing the erroneous sign guard restored the true vignetting boundary ($z_{\mathrm{vig}} \approx 34.04$ m for $128 \times 5\,\mu\text{m}$).

### 3.3 Spatial Resolution Convergence in Thin-Lens Focusing
The thin-lens validation verifies Gaussian beam focusing against Kogelnik & Li (1966) ABCD theory [R4]. Analytical parameters:
$$z_f = \frac{f}{1 + (f / z_R)^2} \approx 73.5356\text{ mm}, \qquad w_0' = \frac{w_0}{\sqrt{1 + (z_R / f)^2}} \approx 41.920\,\mu\text{m}$$
Longitudinal waist scanning with continuous local 3-point quadratic interpolation yields:
- **Quick Mode ($N = 256$, $\Delta x = 10\,\mu\text{m}$):** $z_f = 73.5330$ mm (raw relative error $0.0036\%$, uncertainty $\pm 0.0250$ mm), $w_0' = 41.939\,\mu\text{m}$ (raw relative error $0.0449\%$).
- **Full Mode ($N = 512$, $\Delta x = 5\,\mu\text{m}$):** $z_f = 73.5350$ mm (raw relative error $0.0008\%$, uncertainty $\pm 0.0100$ mm), $w_0' = 41.920\,\mu\text{m}$ (raw relative error $< 0.0001\%$).
The $2.0\,\mu\text{m}$ difference between Quick and Full modes reflects spatial grid resolution convergence, not an implementation contradiction.

---

## 4. Quantitative Validation Results

The table below reports the complete quantitative validation suite from `results/validation.json` generated using the audited codebase (Full Mode):

| Physical Experiment | Method Tested | Distance $z$ | Theoretical Target | Measured Value | Raw Relative Discrepancy | Numerical Uncertainty | Validation Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Zero-distance identity** | Fresnel | $0.0$ m | $U(x, y, 0) = U_0$ | $U_{\mathrm{num}}$ | $2.01 \times 10^{-16}$ | Machine $\epsilon$ | **PASS** |
| **Zero-distance identity** | ASM | $0.0$ m | $U(x, y, 0) = U_0$ | $U_{\mathrm{num}}$ | $2.01 \times 10^{-16}$ | Machine $\epsilon$ | **PASS** |
| **Zero-distance identity** | BL-ASM | $0.0$ m | $U(x, y, 0) = U_0$ | $U_{\mathrm{num}}$ | $2.01 \times 10^{-16}$ | Machine $\epsilon$ | **PASS** |
| **Zero-distance identity** | SAS | $0.0$ m | $U(x, y, 0) = U_0$ | $U_{\mathrm{num}}$ | $0.00 \times 10^{0}$ | Branch Identity | **PASS** |
| **Circular Airy Ring** | BL-ASM | $0.50$ m | $r_1 = 1.0814$ mm | $r_1 = 1.0880$ mm | $0.6065\%$ | $\Delta r = 0.010$ mm | **PASS** |
| **Circular Airy Profile** | BL-ASM | $0.50$ m | $J_1(q)/q$ profile | Radial profile | $1.6234\%$ | Parabolic trunc. | **PASS** |
| **Circular Airy Ring** | SAS | $0.50$ m | $r_1 = 1.0814$ mm | $r_1 = 1.0862$ mm | $0.4370\%$ | $\Delta r = 0.005$ mm | **PASS** |
| **Circular Airy Profile** | SAS | $0.50$ m | $J_1(q)/q$ profile | Radial profile | $0.3585\%$ | Scaled grid area | **PASS** |
| **Rectangular Sinc Slice $x$** | BL-ASM | $0.50$ m | Analytical $\mathrm{sinc}^2(x)$ | Horizontal cross-section | $1.6515\%$ | Grid sampling | **PASS** |
| **Rectangular Sinc Slice $y$** | BL-ASM | $0.50$ m | Analytical $\mathrm{sinc}^2(y)$ | Vertical cross-section | $2.6360\%$ | Grid sampling | **PASS** |
| **Near-Field Direct Integration** | BL-ASM | $0.005$ m | Rayleigh–Sommerfeld direct | Central $80 \times 80$ subgrid | $3.8926\%$ (intensity)<br>$7.8087\%$ (complex) | Quad discretization<br>$\Delta \phi = -0.0034$ rad | **PASS** |
| **Thin-Lens Focus Position** | BL-ASM | Focus | $z_f = 73.5356$ mm | $z_{f,\mathrm{meas}} = 73.5350$ mm | $0.0008\%$ ($8.18 \times 10^{-6}$) | $\pm 0.0100$ mm | **PASS** |
| **Thin-Lens Focused Waist** | BL-ASM | Focus | $w_0' = 41.9199\,\mu$m | $w_{0,\mathrm{meas}}' = 41.9199\,\mu$m | $0.0000\%$ ($4.21 \times 10^{-9}$) | $\pm 0.1000\,\mu$m | **PASS** |
| **Thin-Lens Profile $L_2$** | BL-ASM | $z \in [30, 120]$ mm | ABCD beam envelope | Transverse second moment | $0.0011\%$ | Profile sweep | **PASS** |
| **Vortex Conservation $\ell = 1$** | BL-ASM | $0.1, 0.2, 0.3$ m | $Q = 1.0$ across 3 radii | $Q_{\mathrm{in}} = 1.000, Q_{\mathrm{out}} = 1.000$ | $0.0000\%$ | Phase unwrap | **PASS** |
| **Vortex Conservation $\ell = 2$** | BL-ASM | $0.1, 0.2, 0.3$ m | $Q = 2.0$ across 3 radii | $Q_{\mathrm{in}} = 2.000, Q_{\mathrm{out}} = 2.000$ | $0.0000\%$ | Phase unwrap | **PASS** |
| **Vortex Conservation $\ell = 3$** | BL-ASM | $0.1, 0.2, 0.3$ m | $Q = 3.0$ across 3 radii | $Q_{\mathrm{in}} = 3.000, Q_{\mathrm{out}} = 3.000$ | $0.0000\%$ | Phase unwrap | **PASS** |
| **Vortex Conservation $\ell = -1$** | BL-ASM | $0.1, 0.2, 0.3$ m | $Q = -1.0$ across 3 radii | $Q_{\mathrm{in}} = -1.000, Q_{\mathrm{out}} = -1.000$ | $0.0000\%$ | Phase unwrap | **PASS** |

---

## 5. Automated Testing and CI Quality Infrastructure

### 5.1 Test Suite Expansion
The pytest test suite (`tests/test_diffraction.py`) was expanded to **39 dedicated tests**:
1. `test_regression_issue_a_method_selection_map_isolation`: Verifies that `generate_fig04_method_selection_map` isolates method selection data when `metrics.csv` contains multiple experiment types.
2. `test_regression_issue_b_vortex_no_theory_substitution`: Verifies that vortex charge visualization does not fabricate theoretical charges when measurement dictionaries omit values, and verifies handling of signed negative charges.
3. `test_regression_issue_c_validation_failure_exit_status`: Verifies that a failing validation check in `reproduce.py` triggers `sys.exit(1)` and does not raise `IndexError` when executed in an isolated temporary directory.
4. `test_save_results_empty_metrics_handling`: Verifies that `save_results` writes a valid empty CSV with standard column headers when passed an empty metrics list.
5. `test_validation_failure_preserves_existing_results`: Verifies that validation failure preserves pre-existing benchmark metrics and does not truncate tracked results.
6. `test_grid_and_pad_even_odd_centering`: Verifies exact pixel centering for 10 distinct even, odd, rectangular, and mixed grid configurations across `create_grid` and `pad_field`.
7. `test_sas_validity_nonsquare_2d_vignetting`: Verifies that `check_sas_validity` evaluates both $x$ and $y$ dimensions on non-square grids, enforces $M = \min(M_x, M_y) > 1$, and correctly identifies vignetting boundaries.
8. `test_complex_field_error_global_phase_alignment`: Verifies that `compute_complex_field_error` analytically aligns global constant phase offsets $e^{i\theta}$ to zero error.
9. `test_physical_parameter_validation`: Verifies strict `ValueError` exceptions for zero/negative wavelengths, negative sampling pitches, and non-positive aperture dimensions.

**Result:** 39 of 39 tests pass in ~31 s on CPU with 0 warnings.

### 5.2 Hardened Continuous Integration Workflow
Continuous integration was configured and hardened in `.github/workflows/ci.yml`:
- **Matrix:** Ubuntu (`ubuntu-latest`) and Windows (`windows-latest`) across Python 3.10, 3.11, 3.12, and 3.13.
- **Result Cleanliness Check:** Immediately following `pytest`, `git diff --exit-code -- results/` asserts that running tests did not modify tracked production results.
- **Reproduction & Strict Artifact Verification:** Runs `python reproduce.py --quick` and verifies that all 7 validation categories in `validation.json` pass (`passed == true`) and that `metrics.csv` contains $> 20$ rows with required columns.

---

## 6. Physical Assumptions and Method Limitations

1. **Scalar Diffraction Regime:** All four propagators solve the scalar Helmholtz equation. Polarization vector coupling and near-field subwavelength evanescent interactions are neglected ($D, w_0 \gg \lambda$).
2. **Transfer-Function Aliasing:** Standard ASM aliases beyond $z_c = p N \Delta x_s^2 / \lambda$. BL-ASM suppresses aliasing by hard-bandlimiting high spatial frequencies, which avoids noise but truncates wide diffraction orders when $f_{\lim} < 3/D$.
3. **Scaled Grid Coordinates:** SAS inherently couples the observation window size and grid pitch to distance ($M = \lambda z / (p N \Delta x_s^2)$). It is mathematically invalid for $M = \min(M_x, M_y) \le 1$ and singular at $z = 0$.
4. **Finite-Window Optical Power vs Linear Coverage:** Missing optical power must not be conflated with geometric window coverage; on-axis peaked diffraction fields retain significant power even when the observation window truncates outer diffraction rings.

---

## 7. Conclusion

The second-round audit and repair has eliminated the critical results-truncation vulnerability, corrected SAS vignetting and magnification physics, resolved historical audit inaccuracies, and fully validated the 292-row numerical benchmark suite and publication figures. All 39 automated tests pass deterministically on CPU, and the repository is fully verified and publication-ready.
