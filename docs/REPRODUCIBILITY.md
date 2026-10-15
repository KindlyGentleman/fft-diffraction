# Reproducibility Guide and Environment Specification

## 1. System Requirements and Environment

This research codebase is written in pure Python 3.11+ and is engineered to run deterministically on standard CPU architectures without requiring GPU hardware, proprietary software, or online network access.

### Hardware Specification for Validated Benchmarks
- **Operating System**: Windows 11 (tested on Windows 10/11 x86-64 and Linux x86-64)
- **Processor**: Intel Core / Xeon or AMD Ryzen (x86-64 architecture)
- **Host RAM**: 8 GB minimum (16 GB recommended for $N \ge 2048$ grids)
- **Python Version**: Python 3.11, 3.12, or 3.13 (benchmarked on Python 3.13.7)

### Software Dependencies
```txt
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
scienceplots>=2.1.0
imageio>=2.30.0
imageio-ffmpeg>=0.4.8
pyyaml>=6.0
pytest>=7.0.0
waveprop>=0.0.12
```

---

## 2. Step-by-Step Reproduction Instructions

### Step 1: Install Package in Editable Mode
From the project root directory:
```bash
pip install -e .
```

### Step 2: Run Scientific Validation Suite
Execute the unit and analytical validation tests:
```bash
python scripts/run_validation.py
```
This tests:
- Zero-distance complex identity ($z = 0$)
- Uniform plane-wave phase evolution
- 2D FFT/IFFT roundtrip invertibility
- Circular Fraunhofer Airy disc profile and first zero radius
- Rectangular Fraunhofer sinc cross sections
- Direct Rayleigh-Sommerfeld numerical integration
- Integrated energy conservation
- Baseline cross-check against `waveprop`

All 8 tests must report `PASS`. Detailed outputs are written to `outputs/metrics/validation_results.json`.

### Step 3: Run Experimental Parameter Sweeps
To run the quick sweep (~30-60 seconds on CPU):
```bash
python scripts/run_experiments.py --mode quick
```
To run the comprehensive high-resolution sweep:
```bash
python scripts/run_experiments.py --mode full
```
This writes CSV result tables to `outputs/metrics/` and saves key simulation arrays to `outputs/data/`.

### Step 4: Generate Publication Figures
To regenerate all IEEE-style figures from stored simulation data:
```bash
python scripts/generate_figures.py
```
Output figures are saved in vector PDF, SVG, and 300 DPI raster PNG formats in `outputs/figures/`:
- `fig01_geometry.{pdf,svg,png}`
- `fig02_apertures.{pdf,svg,png}`
- `fig03_method_comparison.{pdf,svg,png}`
- `fig04_analytical_validation.{pdf,svg,png}`
- `fig05_sampling_convergence.{pdf,svg,png}`
- `fig06_distance_study.{pdf,svg,png}`
- `fig07_bandlimiting_analysis.{pdf,svg,png}`
- `fig08_cost_accuracy.{pdf,svg,png}`
- `fig09_failure_case.{pdf,svg,png}`

### Step 5: Generate Animated Video Showcase
To render the draft video (15 seconds, low resolution):
```bash
python scripts/generate_video.py --mode draft
```
To render the full 1080p educational showcase (77 seconds, 1920x1080, 24 FPS):
```bash
python scripts/generate_video.py --mode full
```
Output video is saved to `outputs/video/diffraction_showcase_1080p.mp4`.

### Step 6: Automated One-Command Reproduction
To execute the entire pipeline end-to-end:
```bash
python scripts/reproduce_all.py
```
