import os
import platform
import time
from typing import Tuple, Dict, Any
import numpy as np

def estimate_memory_mb(nx: int, ny: int, padding: int = 2) -> float:
    """Estimate memory requirement in MB for complex128 arrays during propagation."""
    nx_pad = nx * padding
    ny_pad = ny * padding
    bytes_per_complex = 16
    array_bytes = nx_pad * ny_pad * bytes_per_complex
    total_est_bytes = array_bytes * 5
    return float(total_est_bytes / (1024.0 * 1024.0))

def check_memory_budget(nx: int, ny: int, padding: int = 2, max_mb: float = 4096.0) -> bool:
    """Verify that the proposed computation fits within the memory budget."""
    est = estimate_memory_mb(nx, ny, padding)
    return est <= max_mb

def radial_average(
    data_2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    num_bins: int = 200,
    r_max: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally averaged radial profile from 2D data."""
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    if r_max is None:
        r_max = min(np.max(np.abs(x)), np.max(np.abs(y)))

    bins = np.linspace(0.0, r_max, num_bins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    bin_indices = np.digitize(R.ravel(), bins) - 1
    valid = (bin_indices >= 0) & (bin_indices < num_bins)

    counts = np.bincount(bin_indices[valid], minlength=num_bins)
    sums = np.bincount(bin_indices[valid], weights=data_2d.ravel()[valid], minlength=num_bins)

    radial_profile = np.zeros(num_bins, dtype=np.float64)
    nonzero = counts > 0
    radial_profile[nonzero] = sums[nonzero] / counts[nonzero]

    return r_centers, radial_profile

def get_system_info() -> Dict[str, Any]:
    """Gather host execution environment metadata."""
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cpu_count": os.cpu_count(),
    }

class BenchmarkTimer:
    """Context manager for accurate runtime measurements."""

    def __init__(self):
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
